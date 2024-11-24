import (
	"math/rand"
)

func findKthLargest(nums []int, k int) int {
	targetIndex := len(nums) - k
	return quickselect(nums, 0, len(nums)-1, targetIndex)
}

func quickselect(nums []int, left int, right int, k int) int {
	if left == right {
		return nums[left]
	}

	// Randomized pivot selection
	pivotIndex := left + rand.Intn(right-left+1)
	nums[pivotIndex], nums[right] = nums[right], nums[pivotIndex]

	// Partition the array and get the pivot index
	pivotIndex = partition(nums, left, right)

	if pivotIndex == k {
		return nums[pivotIndex]
	} else if pivotIndex < k {
		// Search in the right part
		return quickselect(nums, pivotIndex+1, right, k)
	} else {
		// Search in the left part
		return quickselect(nums, left, pivotIndex-1, k)
	}
}

func partition(nums []int, left int, right int) int {
	pivot := nums[right]
	i := left

	for j := left; j < right; j++ {
		if nums[j] < pivot {
			nums[i], nums[j] = nums[j], nums[i]
			i++
		}
	}

	// Place the pivot in its correct position
	nums[i], nums[right] = nums[right], nums[i]
	return i
}


import (
	"container/heap"
	"sort"
)

func findMaximizedCapital(k int, w int, profits []int, capital []int) int {
	n := len(profits)

	// Combine profits and capital into a list of projects
	projects := make([][2]int, n)
	for i := 0; i < n; i++ {
		projects[i] = [2]int{capital[i], profits[i]}
	}

	// Sort projects by capital required in ascending order
	sort.Slice(projects, func(i, j int) bool {
		return projects[i][0] < projects[j][0]
	})

	// Max-Heap to store profits of affordable projects
	profitHeap := &MaxHeap{}
	heap.Init(profitHeap)

	i := 0
	for k > 0 {
		// Add all projects that can be started with the current capital
		for i < n && projects[i][0] <= w {
			heap.Push(profitHeap, projects[i][1])
			i++
		}

		// If no projects can be started, break
		if profitHeap.Len() == 0 {
			break
		}

		// Pick the most profitable project
		w += heap.Pop(profitHeap).(int)
		k--
	}

	return w
}

// MaxHeap implementation
type MaxHeap []int

func (h MaxHeap) Len() int           { return len(h) }
func (h MaxHeap) Less(i, j int) bool { return h[i] > h[j] } // Max-Heap
func (h MaxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *MaxHeap) Push(x interface{}) {
	*h = append(*h, x.(int))
}

func (h *MaxHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}


import "math"

func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
    // Ensure nums1 is the smaller array
    if len(nums1) > len(nums2) {
        return findMedianSortedArrays(nums2, nums1)
    }

    m, n := len(nums1), len(nums2)
    left, right := 0, m
    halfLen := (m + n + 1) / 2

    for left <= right {
        partition1 := (left + right) / 2
        partition2 := halfLen - partition1

        // Use math.Min/Max for boundary conditions
        maxLeft1 := math.MinInt32
        if partition1 > 0 {
            maxLeft1 = nums1[partition1-1]
        }

        minRight1 := math.MaxInt32
        if partition1 < m {
            minRight1 = nums1[partition1]
        }

        maxLeft2 := math.MinInt32
        if partition2 > 0 {
            maxLeft2 = nums2[partition2-1]
        }

        minRight2 := math.MaxInt32
        if partition2 < n {
            minRight2 = nums2[partition2]
        }

        // Check if we've found the correct partition
        if maxLeft1 <= minRight2 && maxLeft2 <= minRight1 {
            if (m+n)%2 == 0 {
                return float64(max(maxLeft1, maxLeft2)+min(minRight1, minRight2)) / 2.0
            }
            return float64(max(maxLeft1, maxLeft2))
        } else if maxLeft1 > minRight2 {
            right = partition1 - 1
        } else {
            left = partition1 + 1
        }
    }

    panic("Input arrays are not sorted")
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}
