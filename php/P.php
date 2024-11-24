<?php

class Solution {
    /**
     * Find the k-th largest element in an array
     *
     * @param Integer[] $nums
     * @param Integer $k
     * @return Integer
     */
    function findKthLargest($nums, $k) {
        $minHeap = new SplPriorityQueue();
        $minHeap->setExtractFlags(SplPriorityQueue::EXTR_DATA); // Use only the data, not priority

        foreach ($nums as $num) {
            $minHeap->insert($num, -$num); // Min-heap, invert priority with negative value
            if ($minHeap->count() > $k) {
                $minHeap->extract(); // Remove the smallest element when heap size exceeds k
            }
        }

        return $minHeap->top(); // The root of the heap is the k-th largest element
    }
}

// Example usage:
$solution = new Solution();
$nums = [3, 2, 1, 5, 6, 4];
$k = 2;

echo $solution->findKthLargest($nums, $k); // Output: 5


<?php

class Solution {
    /**
     * @param Integer[] $nums1
     * @param Integer[] $nums2
     * @return Float
     */
    function findMedianSortedArrays($nums1, $nums2) {
        $m = count($nums1);
        $n = count($nums2);

        // Ensure nums1 is the smaller array
        if ($m > $n) {
            return $this->findMedianSortedArrays($nums2, $nums1);
        }

        $low = 0;
        $high = $m;
        $half_len = intval(($m + $n + 1) / 2);

        while ($low <= $high) {
            $i = intval(($low + $high) / 2);
            $j = $half_len - $i;

            $nums1_left = ($i == 0) ? PHP_INT_MIN : $nums1[$i - 1];
            $nums1_right = ($i == $m) ? PHP_INT_MAX : $nums1[$i];
            $nums2_left = ($j == 0) ? PHP_INT_MIN : $nums2[$j - 1];
            $nums2_right = ($j == $n) ? PHP_INT_MAX : $nums2[$j];

            if ($nums1_left <= $nums2_right && $nums2_left <= $nums1_right) {
                // Found the correct partition
                if (($m + $n) % 2 == 0) {
                    return (max($nums1_left, $nums2_left) + min($nums1_right, $nums2_right)) / 2.0;
                } else {
                    return max($nums1_left, $nums2_left);
                }
            } elseif ($nums1_left > $nums2_right) {
                $high = $i - 1;
            } else {
                $low = $i + 1;
            }
        }

        throw new Exception("Input arrays are not sorted or invalid");
    }
}

// Example usage:
$solution = new Solution();
$nums1 = [1, 3];
$nums2 = [2];
echo $solution->findMedianSortedArrays($nums1, $nums2); // Output: 2.0

$nums1 = [1, 2];
$nums2 = [3, 4];
echo $solution->findMedianSortedArrays($nums1, $nums2); // Output: 2.5




<?php

class Solution {
    /**
     * @param Integer[] $nums
     * @return Integer
     */
    function findMin($nums) {
        $left = 0;
        $right = count($nums) - 1;

        while ($left < $right) {
            $mid = intval($left + ($right - $left) / 2);

            // If the mid element is greater than the rightmost element, 
            // the minimum must be in the right part of the array.
            if ($nums[$mid] > $nums[$right]) {
                $left = $mid + 1;
            } else {
                // Otherwise, the minimum is in the left part (including mid).
                $right = $mid;
            }
        }

        // When the loop exits, left == right, pointing to the minimum element
        return $nums[$left];
    }
}

// Example usage:
$solution = new Solution();

// Example 1
$nums = [3, 4, 5, 1, 2];
echo $solution->findMin($nums) . "\n"; // Output: 1

// Example 2
$nums = [4, 5, 6, 7, 0, 1, 2];
echo $solution->findMin($nums) . "\n"; // Output: 0

// Example 3
$nums = [11, 13, 15, 17];
echo $solution->findMin($nums) . "\n"; // Output: 11
