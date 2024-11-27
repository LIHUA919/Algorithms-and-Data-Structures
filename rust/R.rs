impl Solution {
    pub fn find_median_sorted_arrays(nums1: Vec<i32>, nums2: Vec<i32>) -> f64 {
        let (mut a, mut b) = if nums1.len() > nums2.len() {
            (nums2, nums1)
        } else {
            (nums1, nums2)
        };

        let (m, n) = (a.len(), b.len());
        let half_len = (m + n + 1) / 2;

        let (mut left, mut right) = (0, m);
        while left <= right {
            let i = (left + right) / 2;
            let j = half_len - i;

            let a_left = if i == 0 { i32::MIN } else { a[i - 1] };
            let a_right = if i == m { i32::MAX } else { a[i] };
            let b_left = if j == 0 { i32::MIN } else { b[j - 1] };
            let b_right = if j == n { i32::MAX } else { b[j] };

            if a_left <= b_right && b_left <= a_right {
                if (m + n) % 2 == 0 {
                    return ((a_left.max(b_left) + a_right.min(b_right)) as f64) / 2.0;
                } else {
                    return a_left.max(b_left) as f64;
                }
            } else if a_left > b_right {
                right = i - 1;
            } else {
                left = i + 1;
            }
        }

        unreachable!()
    }
}



impl Solution {
    pub fn find_min(nums: Vec<i32>) -> i32 {
        let mut left = 0;
        let mut right = nums.len() - 1;

        while left < right {
            let mid = left + (right - left) / 2;

            // If the middle element is greater than the rightmost element,
            // the minimum is in the right half.
            if nums[mid] > nums[right] {
                left = mid + 1;
            } else {
                // Otherwise, the minimum is in the left half (including mid).
                right = mid;
            }
        }

        // When the loop exits, left == right, pointing to the minimum element.
        nums[left]
    }
}



impl Solution {
    pub fn search_range(nums: Vec<i32>, target: i32) -> Vec<i32> {
        let first = Self::binary_search(&nums, target, true);
        if first == -1 {
            return vec![-1, -1]; // Target not found
        }
        let last = Self::binary_search(&nums, target, false);
        vec![first, last]
    }

    fn binary_search(nums: &[i32], target: i32, find_first: bool) -> i32 {
        let mut left = 0;
        let mut right = nums.len() as i32 - 1;
        let mut result = -1;

        while left <= right {
            let mid = left + (right - left) / 2;
            if nums[mid as usize] == target {
                result = mid; // Potential match
                if find_first {
                    right = mid - 1; // Narrow search to left
                } else {
                    left = mid + 1; // Narrow search to right
                }
            } else if nums[mid as usize] < target {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        result
    }
}

impl Solution {
    pub fn search(nums: Vec<i32>, target: i32) -> i32 {
        let mut left = 0;
        let mut right = nums.len() as i32 - 1;

        while left <= right {
            let mid = left + (right - left) / 2;

            if nums[mid as usize] == target {
                return mid;
            }

            // Determine if the left half is sorted
            if nums[left as usize] <= nums[mid as usize] {
                // Target is in the left sorted portion
                if nums[left as usize] <= target && target < nums[mid as usize] {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                // The right half is sorted
                if nums[mid as usize] < target && target <= nums[right as usize] {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }

        -1 // Target not found
    }
}


impl Solution {
    pub fn find_peak_element(nums: Vec<i32>) -> i32 {
        let mut left = 0;
        let mut right = nums.len() - 1;

        while left < right {
            let mid = left + (right - left) / 2;

            if nums[mid] > nums[mid + 1] {
                // There is a peak in the left half
                right = mid;
            } else {
                // There is a peak in the right half
                left = mid + 1;
            }
        }

        left as i32
    }
}


impl Solution {
    pub fn search_matrix(matrix: Vec<Vec<i32>>, target: i32) -> bool {
        let m = matrix.len();
        let n = matrix[0].len();
        let mut left: usize = 0;
        let mut right: usize = m * n;

        while left < right {
            // Calculate the middle index
            let mid = left + (right - left) / 2;
            let row = mid / n;
            let col = mid % n;

            let mid_value = matrix[row][col];
            if mid_value == target {
                return true;
            } else if mid_value < target {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        false
    }
}













impl Solution {
    pub fn search_insert(nums: Vec<i32>, target: i32) -> i32 {
        let mut left = 0;
        let mut right = nums.len() as i32 - 1;

        while left <= right {
            let mid = left + (right - left) / 2;

            if nums[mid as usize] == target {
                return mid;
            } else if nums[mid as usize] < target {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        left
    }
}
