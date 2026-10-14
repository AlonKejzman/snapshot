// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package criu

import (
	"slices"
	"testing"

	"github.com/checkpoint-restore/go-criu/v8/crit/images/fdinfo"
	"github.com/checkpoint-restore/go-criu/v8/crit/images/fown"
	"github.com/checkpoint-restore/go-criu/v8/crit/images/regfile"
	"golang.org/x/sys/unix"
	"google.golang.org/protobuf/proto"
)

func TestReferencedPaths(t *testing.T) {
	dir := t.TempDir()
	writeFilesImage(t, dir, []*fdinfo.FileEntry{
		newRegFileEntry(1, "/root/.triton/cache/K/cuda_utils.so"),
		newRegFileEntry(2, "/usr/lib/x86_64-linux-gnu/libc.so.6"),
		// Every process that maps the same file gets its own entry.
		newRegFileEntry(3, "/root/.triton/cache/K/cuda_utils.so"),
		newRegFileEntry(4, "/vllm-workspace/"),
		newRegFileEntry(5, "relative/name"),
		newUnixSocketEntry(6, []byte("/tmp/sock\x00"), 106, unix.SOCK_STREAM, linuxUnixSocketStateListen),
	})

	got, err := ReferencedPaths(dir)
	if err != nil {
		t.Fatalf("ReferencedPaths: %v", err)
	}
	want := []string{
		"/root/.triton/cache/K/cuda_utils.so",
		"/usr/lib/x86_64-linux-gnu/libc.so.6",
		"/vllm-workspace",
	}
	if !slices.Equal(got, want) {
		t.Fatalf("ReferencedPaths() = %q, want %q", got, want)
	}
}

func TestReferencedPathsRequiresFilesImage(t *testing.T) {
	if _, err := ReferencedPaths(t.TempDir()); err == nil {
		t.Fatal("ReferencedPaths should fail without files.img")
	}
}

func newRegFileEntry(id uint32, name string) *fdinfo.FileEntry {
	zero32 := uint32(0)
	return &fdinfo.FileEntry{
		Type: fdinfo.FdTypes_REG.Enum(),
		Id:   proto.Uint32(id),
		Reg: &regfile.RegFileEntry{
			Id:    proto.Uint32(id),
			Flags: &zero32,
			Pos:   proto.Uint64(0),
			Fown: &fown.FownEntry{
				Uid:     &zero32,
				Euid:    &zero32,
				Signum:  &zero32,
				PidType: &zero32,
				Pid:     &zero32,
			},
			Name: proto.String(name),
		},
	}
}
