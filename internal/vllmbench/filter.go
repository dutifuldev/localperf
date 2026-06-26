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
	if len(profileFilter) == 0 && len(workloadFilter) == 0 && len(concurrencyFilter) == 0 {
		return nil
	}
	if len(profileFilter) > 0 {
		profiles := spec.Profiles[:0]
		for _, profile := range spec.Profiles {
			if profileFilter[profile.Name] {
				profiles = append(profiles, profile)
			}
		}
		spec.Profiles = profiles
	}
	if len(workloadFilter) > 0 {
		workloads := spec.Workloads[:0]
		for _, workload := range spec.Workloads {
			if workloadFilter[workload.Name] {
				workloads = append(workloads, workload)
			}
		}
		spec.Workloads = workloads
	}
	if len(profileFilter) > 0 {
		for i := range spec.Workloads {
			workload := &spec.Workloads[i]
			if len(workload.Profiles) == 0 {
				workload.Profiles = filter.Profiles
				continue
			}
			profiles := workload.Profiles[:0]
			for _, profileName := range workload.Profiles {
				if profileFilter[profileName] {
					profiles = append(profiles, profileName)
				}
			}
			workload.Profiles = profiles
		}
	}
	if len(concurrencyFilter) > 0 {
		for i := range spec.Workloads {
			workload := &spec.Workloads[i]
			values := workload.MaxConcurrency[:0]
			for _, concurrency := range workload.MaxConcurrency {
				if concurrencyFilter[concurrency] {
					values = append(values, concurrency)
				}
			}
			workload.MaxConcurrency = values
		}
	}
	if err := ValidateSpec(*spec); err != nil {
		return fmt.Errorf("filter produced invalid spec: %w", err)
	}
	return nil
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
