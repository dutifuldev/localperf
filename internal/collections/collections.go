package collections

import (
	"cmp"
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"
)

func SortedKeys[M ~map[K]V, K cmp.Ordered, V any](values M) []K {
	return slices.Sorted(maps.Keys(values))
}

func JoinInts(values []int, sep string) string {
	parts := make([]string, 0, len(values))
	for _, value := range values {
		parts = append(parts, fmt.Sprint(value))
	}
	return strings.Join(parts, sep)
}

func JoinIntKeys[V any](values map[int]V, sep string) string {
	return JoinInts(SortedKeys(values), sep)
}

type PositiveIntList []int

func (values *PositiveIntList) String() string {
	if values == nil {
		return ""
	}
	return JoinInts(*values, ",")
}

func (values *PositiveIntList) Set(raw string) error {
	value, err := strconv.Atoi(raw)
	if err != nil || value <= 0 {
		return fmt.Errorf("invalid positive integer %q", raw)
	}
	*values = append(*values, value)
	return nil
}
