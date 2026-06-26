package vllmbench

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

type MemorySnapshot struct {
	MemTotalGiB     float64 `json:"mem_total_gib"`
	MemAvailableGiB float64 `json:"mem_available_gib"`
	SwapFreeGiB     float64 `json:"swap_free_gib"`
}

type MemoryFloorError struct {
	Snapshot MemorySnapshot
	MinGiB   float64
}

func (err *MemoryFloorError) Error() string {
	return fmt.Sprintf("MemAvailable %.1f GiB is below floor %.1f GiB", err.Snapshot.MemAvailableGiB, err.MinGiB)
}

func IsMemoryFloorError(err error) bool {
	var floorErr *MemoryFloorError
	return errors.As(err, &floorErr)
}

var checkMemoryFloor = CheckMemoryFloor

func ReadMemorySnapshot() (MemorySnapshot, error) {
	file, err := os.Open("/proc/meminfo")
	if err != nil {
		return MemorySnapshot{}, err
	}
	defer file.Close()
	return ParseMeminfo(file)
}

func ParseMeminfo(reader io.Reader) (MemorySnapshot, error) {
	values := map[string]float64{}
	scanner := bufio.NewScanner(reader)
	for scanner.Scan() {
		line := scanner.Text()
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		key := strings.TrimSuffix(fields[0], ":")
		kib, err := strconv.ParseFloat(fields[1], 64)
		if err != nil {
			continue
		}
		values[key] = kib / 1024 / 1024
	}
	if err := scanner.Err(); err != nil {
		return MemorySnapshot{}, err
	}
	if values["MemAvailable"] <= 0 {
		return MemorySnapshot{}, fmt.Errorf("MemAvailable not found in meminfo")
	}
	return MemorySnapshot{
		MemTotalGiB:     values["MemTotal"],
		MemAvailableGiB: values["MemAvailable"],
		SwapFreeGiB:     values["SwapFree"],
	}, nil
}

func CheckMemoryFloor(minGiB float64) (MemorySnapshot, error) {
	snapshot, err := ReadMemorySnapshot()
	if err != nil {
		return snapshot, err
	}
	if snapshot.MemAvailableGiB < minGiB {
		return snapshot, &MemoryFloorError{Snapshot: snapshot, MinGiB: minGiB}
	}
	return snapshot, nil
}
