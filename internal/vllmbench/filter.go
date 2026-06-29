package vllmbench

import "fmt"

type Filter struct {
	Profiles      []string
	Workloads     []string
	Concurrencies []int
}

func ApplyFilter(spec *Spec, filter Filter) error {
	profileFilter := stringSet(filter.Profiles)
	workloadFilter := stringSet(filter.Workloads)
	concurrencyFilter := intSet(filter.Concurrencies)
	if filtersEmpty(profileFilter, workloadFilter, concurrencyFilter) {
		return nil
	}
	if len(profileFilter) > 0 {
		spec.Profiles = filterProfiles(spec.Profiles, profileFilter)
	}
	if len(workloadFilter) > 0 {
		spec.Workloads = filterWorkloadsByName(spec.Workloads, workloadFilter)
	}
	if len(profileFilter) > 0 {
		spec.Workloads = filterWorkloadsByProfiles(spec.Workloads, filter.Profiles, profileFilter)
	}
	if len(concurrencyFilter) > 0 {
		spec.Workloads = filterWorkloadsByConcurrency(spec.Workloads, concurrencyFilter)
	}
	if err := ValidateSpec(*spec); err != nil {
		return fmt.Errorf("filter produced invalid spec: %w", err)
	}
	return nil
}

func filtersEmpty(profileFilter, workloadFilter map[string]bool, concurrencyFilter map[int]bool) bool {
	return len(profileFilter) == 0 && len(workloadFilter) == 0 && len(concurrencyFilter) == 0
}

func filterProfiles(profiles []Profile, filter map[string]bool) []Profile {
	return filterSlice(profiles, func(profile Profile) bool { return filter[profile.Name] })
}

func filterWorkloadsByName(workloads []Workload, filter map[string]bool) []Workload {
	return filterSlice(workloads, func(workload Workload) bool { return filter[workload.Name] })
}

func filterWorkloadsByProfiles(workloads []Workload, requested []string, filter map[string]bool) []Workload {
	out := workloads[:0]
	for _, workload := range workloads {
		profiles, keep := filteredWorkloadProfiles(workload.Profiles, requested, filter)
		if keep {
			workload.Profiles = profiles
			out = append(out, workload)
		}
	}
	return out
}

func filteredWorkloadProfiles(profiles, requested []string, filter map[string]bool) ([]string, bool) {
	if len(profiles) == 0 {
		return requested, true
	}
	selected := filterSlice(profiles, func(profile string) bool { return filter[profile] })
	return selected, len(selected) > 0
}

func filterWorkloadsByConcurrency(workloads []Workload, filter map[int]bool) []Workload {
	out := workloads[:0]
	for _, workload := range workloads {
		workload.MaxConcurrency = filterSlice(workload.MaxConcurrency, func(value int) bool { return filter[value] })
		if len(workload.MaxConcurrency) > 0 {
			out = append(out, workload)
		}
	}
	return out
}

func filterSlice[T any](values []T, keep func(T) bool) []T {
	out := values[:0]
	for _, value := range values {
		if keep(value) {
			out = append(out, value)
		}
	}
	return out
}

func stringSet(values []string) map[string]bool {
	out := map[string]bool{}
	for _, value := range values {
		if value != "" {
			out[value] = true
		}
	}
	return out
}

func intSet(values []int) map[int]bool {
	out := map[int]bool{}
	for _, value := range values {
		if value > 0 {
			out[value] = true
		}
	}
	return out
}
