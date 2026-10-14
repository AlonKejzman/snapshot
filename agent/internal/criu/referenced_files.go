// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package criu

import (
	"fmt"
	"path"
	"slices"

	"github.com/checkpoint-restore/go-criu/v8/crit/images/fdinfo"
)

// ReferencedPaths returns the absolute container paths of the regular files
// and directories a dump references: open descriptors, file-backed mappings,
// executables, working and root directories, and the link-remap names CRIU
// created for unlinked files. Restore re-opens every one of them by path.
func ReferencedPaths(checkpointDir string) ([]string, error) {
	image, err := decodeFilesImage(checkpointDir)
	if err != nil {
		return nil, err
	}
	var paths []string
	for i, entry := range image.Entries {
		fileEntry, ok := entry.Message.(*fdinfo.FileEntry)
		if !ok {
			return nil, fmt.Errorf("unexpected %s entry %d type %T", filesImageFilename, i, entry.Message)
		}
		if fileEntry.GetType() != fdinfo.FdTypes_REG || fileEntry.Reg == nil {
			continue
		}
		if name := fileEntry.Reg.GetName(); path.IsAbs(name) {
			paths = append(paths, path.Clean(name))
		}
	}
	slices.Sort(paths)
	return slices.Compact(paths), nil
}
