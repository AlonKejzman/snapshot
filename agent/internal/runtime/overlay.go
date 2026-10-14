// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package runtime

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"

	"github.com/go-logr/logr"

	"github.com/ai-dynamo/snapshot/agent/internal/nsmount"
	"github.com/ai-dynamo/snapshot/agent/internal/types"
)

const (
	rootfsDiffFilename   = "rootfs-diff.tar"
	deletedFilesFilename = "deleted-files.json"
)

// GetRootFS returns the container's root filesystem path via /host/proc.
func GetRootFS(pid int) (string, error) {
	rootPath := fmt.Sprintf("%s/%d/root", HostProcPath, pid)
	if _, err := os.Stat(rootPath); err != nil {
		return "", fmt.Errorf("rootfs not accessible at %s: %w", rootPath, err)
	}
	return rootPath, nil
}

// GetOverlayUpperDir extracts the overlay upperdir from mountinfo.
func GetOverlayUpperDir(pid int) (string, error) {
	mountInfo, err := ReadMountInfo(pid)
	if err != nil {
		return "", fmt.Errorf("failed to parse mountinfo: %w", err)
	}

	for _, mount := range mountInfo {
		if mount.MountPoint != "/" || mount.FSType != "overlay" {
			continue
		}

		for _, opt := range strings.Split(mount.VFSOptions, ",") {
			if strings.HasPrefix(opt, "upperdir=") {
				return strings.TrimPrefix(opt, "upperdir="), nil
			}
		}
	}

	return "", fmt.Errorf("overlay upperdir not found for pid %d", pid)
}

// CaptureRootfsDiff captures the overlay upperdir to a tar file.
//
// referenced lists the container paths the checkpointed processes hold open or
// mapped (criu.ReferencedPaths). Restore re-opens each one by path, so any of
// them found in the upperdir is archived even when an exclusion matches it:
// exclusions can trim regenerable state such as compiler caches without
// dropping a file the restored processes need.
//
// The archive is written to a temporary file first and renamed atomically on
// success, so a partial or failed capture never leaves a corrupt archive at the
// final path.
func CaptureRootfsDiff(upperDir, checkpointDir string, exclusions types.OverlaySettings, bindMountDests, referenced []string, log logr.Logger) (string, error) {
	if upperDir == "" {
		return "", fmt.Errorf("upperdir is empty")
	}

	rootfsDiffPath := filepath.Join(checkpointDir, rootfsDiffFilename)

	tmpFile, err := os.CreateTemp(checkpointDir, rootfsDiffFilename+".*.tmp")
	if err != nil {
		return "", fmt.Errorf("failed to create temporary rootfs diff: %w", err)
	}
	tmpPath := tmpFile.Name()
	if err := tmpFile.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return "", fmt.Errorf("failed to close temporary rootfs diff: %w", err)
	}
	defer func() {
		if tmpPath != "" {
			_ = os.Remove(tmpPath)
		}
	}()

	kept := referencedUpperPaths(upperDir, bindMountDests, referenced)
	args := captureTarArgs(upperDir, tmpPath, exclusions, bindMountDests)
	if len(kept) > 0 {
		// The member listing tells which referenced paths the exclusions dropped.
		args = append([]string{"--verbose", "--quoting-style=literal"}, args...)
	}
	var listing, stderr bytes.Buffer
	cmd := exec.Command("tar", args...)
	cmd.Stdout = &listing
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("tar failed: %w (output: %s)", err, stderr.String())
	}

	if members, files := unarchivedMembers(kept, listing.String()); len(members) > 0 {
		cmd := exec.Command("tar", appendTarArgs(upperDir, tmpPath)...)
		cmd.Stdin = strings.NewReader(strings.Join(members, "\x00") + "\x00")
		if output, err := cmd.CombinedOutput(); err != nil {
			return "", fmt.Errorf("tar append of referenced files failed: %w (output: %s)", err, string(output))
		}
		log.Info("Kept referenced files matched by rootfs exclusions", "count", len(files), "paths", firstN(files, 20))
	}

	info, err := os.Stat(tmpPath)
	if err != nil {
		return "", fmt.Errorf("failed to stat temporary rootfs diff: %w", err)
	}
	if info.Size() == 0 {
		return "", fmt.Errorf("tar produced empty rootfs diff")
	}

	if err := os.Rename(tmpPath, rootfsDiffPath); err != nil {
		return "", fmt.Errorf("failed to publish rootfs diff: %w", err)
	}
	tmpPath = ""

	return rootfsDiffPath, nil
}

// captureTarArgs builds the archive-creation arguments for a rootfs diff.
//
// trusted.* is excluded because the archive is cut from the overlay upperdir:
// it carries trusted.overlay.opaque markers, and re-applying one to the
// restored container hides the base-image content under it.
func captureTarArgs(upperDir, outPath string, exclusions types.OverlaySettings, bindMountDests []string) []string {
	args := []string{"--xattrs", "--xattrs-exclude=trusted.*"}
	for _, excl := range buildExclusions(exclusions) {
		args = append(args, "--exclude="+excl)
	}
	for _, dest := range bindMountDests {
		args = append(args, "--exclude=."+dest)
	}
	return append(args, "-C", upperDir, "-cf", outPath, ".")
}

// appendTarArgs builds the arguments that append the NUL-separated members
// read from stdin to an archive captureTarArgs created, with the same xattr
// handling and without descending into directories.
func appendTarArgs(upperDir, outPath string) []string {
	return []string{
		"--xattrs", "--xattrs-exclude=trusted.*", "--no-recursion",
		"-C", upperDir, "-rf", outPath, "--null", "--verbatim-files-from", "-T", "-",
	}
}

// referencedUpperPaths returns, relative to the upperdir and sorted, the
// referenced paths that are regular files or directories in the upperdir.
// Paths under bind mounts and kernel filesystems are skipped: the restore
// container provides those, and the upperdir can only hold stale copies shadowed
// by the mount. A path is also skipped when an upperdir ancestor is not a
// directory, so archiving it by name never resolves through a symlink.
func referencedUpperPaths(upperDir string, bindMountDests, referenced []string) []string {
	skipped := append([]string{"/proc", "/sys", "/dev"}, bindMountDests...)
	var kept []string
	for _, p := range referenced {
		p = filepath.Clean(p)
		if !filepath.IsAbs(p) || p == "/" || slices.ContainsFunc(skipped, func(mount string) bool { return underPath(p, mount) }) {
			continue
		}
		rel := strings.TrimPrefix(p, "/")
		if isUpperDirEntry(upperDir, rel) {
			kept = append(kept, rel)
		}
	}
	slices.Sort(kept)
	return slices.Compact(kept)
}

// isUpperDirEntry reports whether rel is a regular file or directory in the
// upperdir, reached through real directories only.
func isUpperDirEntry(upperDir, rel string) bool {
	current := upperDir
	parts := strings.Split(rel, "/")
	for _, part := range parts[:len(parts)-1] {
		current = filepath.Join(current, part)
		if info, err := os.Lstat(current); err != nil || !info.IsDir() {
			return false
		}
	}
	info, err := os.Lstat(filepath.Join(current, parts[len(parts)-1]))
	return err == nil && (info.Mode().IsRegular() || info.IsDir())
}

// unarchivedMembers returns the archive members to append so every kept path
// is in the archive: each kept path, preceded by its ancestors, that the
// listing of the created archive does not name. files holds the kept paths
// among them. Ancestors are archived too so the restored directories keep
// their modes and owners. A name the literal listing cannot represent, such as
// one containing a newline, is appended again; extraction skips the duplicate.
func unarchivedMembers(kept []string, listing string) (members, files []string) {
	archived := make(map[string]struct{})
	for _, line := range strings.Split(listing, "\n") {
		archived[strings.TrimSuffix(strings.TrimPrefix(line, "./"), "/")] = struct{}{}
	}
	for _, rel := range kept {
		parts := strings.Split(rel, "/")
		for i := range parts {
			name := strings.Join(parts[:i+1], "/")
			if _, ok := archived[name]; ok {
				continue
			}
			archived[name] = struct{}{}
			members = append(members, "./"+name)
			if name == rel {
				files = append(files, "/"+rel)
			}
		}
	}
	return members, files
}

func underPath(p, root string) bool {
	root = filepath.Clean(root)
	return p == root || strings.HasPrefix(p, root+"/")
}

func firstN(values []string, n int) []string {
	if len(values) <= n {
		return values
	}
	return values[:n]
}

// buildExclusions merges exclusion lists and normalizes paths for tar --exclude patterns.
func buildExclusions(s types.OverlaySettings) []string {
	exclusions := append([]string(nil), s.Exclusions...)
	for i, p := range exclusions {
		if strings.HasPrefix(p, "*") {
			continue
		}
		p = strings.TrimPrefix(p, ".")
		p = strings.TrimPrefix(p, "/")
		exclusions[i] = "./" + p
	}
	return exclusions
}

// CaptureDeletedFiles finds whiteout files and saves them to a JSON file.
func CaptureDeletedFiles(upperDir, checkpointDir string) (bool, error) {
	if upperDir == "" {
		return false, nil
	}

	whiteouts, err := findWhiteoutFiles(upperDir)
	if err != nil {
		return false, fmt.Errorf("failed to find whiteout files: %w", err)
	}

	if len(whiteouts) == 0 {
		return false, nil
	}

	deletedFilesPath := filepath.Join(checkpointDir, deletedFilesFilename)
	data, err := json.Marshal(whiteouts)
	if err != nil {
		return false, fmt.Errorf("failed to marshal whiteouts: %w", err)
	}

	if err := os.WriteFile(deletedFilesPath, data, 0644); err != nil {
		return false, fmt.Errorf("failed to write deleted files: %w", err)
	}

	return true, nil
}

// restoreTarCmd builds the extraction command for a rootfs diff.
//
// The bundled tar runs under the bundled loader, named explicitly so the
// kernel never consults its PT_INTERP and resolves libc from the placeholder.
// See the bundle layout comment in internal/nsmount for why that is necessary
// and why the loader's glibc is kept out of lib/.
//
// bundleDir must be absolute. There is deliberately no PATH fallback: resolving
// "tar" inside the placeholder would find the placeholder's own tar, which is
// the failure this function exists to prevent.
func restoreTarCmd(bundleDir string, args ...string) (*exec.Cmd, error) {
	if !filepath.IsAbs(bundleDir) {
		return nil, fmt.Errorf("bundle dir must be an absolute path, got %q", bundleDir)
	}
	libcDir := filepath.Join(bundleDir, nsmount.BundleLibcDir)
	argv := append([]string{
		"--library-path", libcDir + ":" + filepath.Join(bundleDir, nsmount.BundleLibDir),
		filepath.Join(bundleDir, nsmount.BundleTar),
	}, args...)
	return exec.Command(filepath.Join(libcDir, nsmount.BundleLoader), argv...), nil
}

// ApplyRootfsDiff extracts rootfs-diff.tar into the target root.
//
// The archive is copied to local disk first. tar walks members with many small
// reads; doing that directly from NFS is much slower than one sequential copy
// plus a local extract.
func ApplyRootfsDiff(checkpointPath, targetRoot, bundleDir string, log logr.Logger) error {
	rootfsDiffPath := filepath.Join(checkpointPath, rootfsDiffFilename)
	info, err := os.Stat(rootfsDiffPath)
	if os.IsNotExist(err) {
		log.V(1).Info("No rootfs-diff.tar, skipping")
		return nil
	}
	if err != nil {
		return fmt.Errorf("failed to stat rootfs diff: %w", err)
	}
	if info.Size() == 0 {
		log.V(1).Info("rootfs-diff.tar is empty, skipping")
		return nil
	}

	localPath, cleanup, err := stageRootfsDiffLocally(rootfsDiffPath)
	if err != nil {
		return err
	}
	defer cleanup()

	// --skip-old-files: silently skip files that already exist in the restore target.
	// The rootfs diff only contains overlay upperdir changes (runtime-generated files
	// like triton caches, tmp files) — base image files should not be overwritten.
	// --numeric-owner: the restored processes hold the checkpoint's numeric ids;
	// re-resolving the archived names through the placeholder's passwd database
	// would remap them.
	// --xattrs-include: tar's extract mask defaults to user.*, so file
	// capabilities would be archived and then dropped. The list is enumerated
	// rather than widened to '*': that would also reapply security.selinux,
	// which can fail the restore on an SELinux node, and the trusted.overlay.*
	// markers carried by archives captured before they were excluded.
	log.Info("Applying rootfs diff", "target", targetRoot, "bytes", info.Size())
	cmd, err := restoreTarCmd(bundleDir,
		"--xattrs", "--xattrs-include=user.*", "--xattrs-include=security.capability",
		"--numeric-owner", "--skip-old-files", "--blocking-factor=2048",
		"-C", targetRoot, "-xf", localPath)
	if err != nil {
		return err
	}
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("tar extract failed: %w", err)
	}
	return nil
}

func stageRootfsDiffLocally(src string) (string, func(), error) {
	in, err := os.Open(src)
	if err != nil {
		return "", nil, fmt.Errorf("failed to open rootfs diff: %w", err)
	}
	defer in.Close()

	tmp, err := os.CreateTemp("", rootfsDiffFilename+".*.tmp")
	if err != nil {
		return "", nil, fmt.Errorf("failed to create local rootfs diff: %w", err)
	}
	tmpPath := tmp.Name()
	cleanup := func() { _ = os.Remove(tmpPath) }

	if _, err := io.Copy(tmp, in); err != nil {
		_ = tmp.Close()
		cleanup()
		return "", nil, fmt.Errorf("failed to stage rootfs diff locally: %w", err)
	}
	if err := tmp.Close(); err != nil {
		cleanup()
		return "", nil, fmt.Errorf("failed to close local rootfs diff: %w", err)
	}
	return tmpPath, cleanup, nil
}

// ApplyDeletedFiles removes files marked as deleted in the checkpoint.
func ApplyDeletedFiles(checkpointPath, targetRoot string, log logr.Logger) error {
	deletedFilesPath := filepath.Join(checkpointPath, deletedFilesFilename)
	data, err := os.ReadFile(deletedFilesPath)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("failed to read deleted files: %w", err)
	}

	var deletedFiles []string
	if err := json.Unmarshal(data, &deletedFiles); err != nil {
		return fmt.Errorf("failed to parse deleted files: %w", err)
	}

	count := 0
	targetRootAbs, err := filepath.Abs(targetRoot)
	if err != nil {
		return fmt.Errorf("failed to resolve target root %s: %w", targetRoot, err)
	}
	targetRootPrefix := targetRootAbs + string(os.PathSeparator)
	for _, f := range deletedFiles {
		if f == "" {
			continue
		}
		target := filepath.Join(targetRoot, f)
		targetAbs, err := filepath.Abs(target)
		if err != nil || (targetAbs != targetRootAbs && !strings.HasPrefix(targetAbs, targetRootPrefix)) {
			log.V(1).Info("Skipping out-of-root deleted file entry", "entry", f)
			continue
		}
		if _, err := os.Stat(target); os.IsNotExist(err) {
			continue
		} else if err != nil {
			log.V(1).Info("Could not stat deleted file target", "path", target, "error", err)
			continue
		}
		if err := os.RemoveAll(target); err != nil {
			log.V(1).Info("Could not delete file", "path", target, "error", err)
			continue
		}
		count++
	}
	log.Info("Deleted files applied", "count", count)
	return nil
}

// findWhiteoutFiles finds overlay whiteout files in the upperdir.
func findWhiteoutFiles(upperDir string) ([]string, error) {
	var whiteouts []string

	err := filepath.Walk(upperDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		name := info.Name()
		if strings.HasPrefix(name, ".wh.") {
			relPath, err := filepath.Rel(upperDir, path)
			if err != nil {
				return fmt.Errorf("failed to compute relative path for %s: %w", path, err)
			}
			dir := filepath.Dir(relPath)
			deletedFile := strings.TrimPrefix(name, ".wh.")
			deletedPath := deletedFile
			if dir != "." {
				deletedPath = filepath.Join(dir, deletedFile)
			}
			whiteouts = append(whiteouts, deletedPath)
		}
		return nil
	})

	return whiteouts, err
}
