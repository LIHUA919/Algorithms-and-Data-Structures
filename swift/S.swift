class Solution {
    func findMin(_ nums: [Int]) -> Int {
        var left = 0
        var right = nums.count - 1

        while left < right {
            let mid = left + (right - left) / 2

            // If the mid element is greater than the rightmost element,
            // the minimum must be in the right part of the array.
            if nums[mid] > nums[right] {
                left = mid + 1
            } else {
                // Otherwise, the minimum is in the left part (including mid).
                right = mid
            }
        }

        // When the loop exits, left == right, pointing to the minimum element.
        return nums[left]
    }
}


class Solution {
    func findMedianSortedArrays(_ nums1: [Int], _ nums2: [Int]) -> Double {
        // Ensure nums1 is the smaller array
        var nums1 = nums1, nums2 = nums2
        if nums1.count > nums2.count {
            swap(&nums1, &nums2)
        }
        
        let m = nums1.count
        let n = nums2.count
        var low = 0, high = m
        
        while low <= high {
            let partitionX = (low + high) / 2
            let partitionY = (m + n + 1) / 2 - partitionX
            
            // Handle boundaries
            let maxX = (partitionX == 0) ? Int.min : nums1[partitionX - 1]
            let minX = (partitionX == m) ? Int.max : nums1[partitionX]
            let maxY = (partitionY == 0) ? Int.min : nums2[partitionY - 1]
            let minY = (partitionY == n) ? Int.max : nums2[partitionY]
            
            if maxX <= minY && maxY <= minX {
                // We have the correct partition
                if (m + n) % 2 == 0 {
                    return Double(max(maxX, maxY) + min(minX, minY)) / 2.0
                } else {
                    return Double(max(maxX, maxY))
                }
            } else if maxX > minY {
                // Move left in nums1
                high = partitionX - 1
            } else {
                // Move right in nums1
                low = partitionX + 1
            }
        }
        
        // If the input arrays are invalid
        fatalError("Input arrays are not sorted or invalid")
    }
}



import Foundation

class Solution {
    func findKthLargest(_ nums: [Int], _ k: Int) -> Int {
        // Min-Heap to track the k largest elements
        var heap = Heap<Int>(sort: <) // Min-Heap
        for num in nums {
            heap.insert(num)
            // Keep the size of the heap at most k
            if heap.count > k {
                heap.remove()
            }
        }
        // The root of the heap is the k-th largest element
        return heap.peek()!
    }
}




// ???? mini heap
// Generic Heap Implementation
struct Heap<Element: Comparable> {
    private var elements: [Element] = []
    private let sort: (Element, Element) -> Bool

    init(sort: @escaping (Element, Element) -> Bool) {
        self.sort = sort
    }

    var count: Int { elements.count }

    func peek() -> Element? { elements.first }

    mutating func insert(_ value: Element) {
        elements.append(value)
        siftUp(from: elements.count - 1)
    }

    mutating func remove() -> Element? {
        guard !elements.isEmpty else { return nil }
        if elements.count == 1 {
            return elements.removeLast()
        } else {
            let value = elements.first
            elements[0] = elements.removeLast()
            siftDown(from: 0)
            return value
        }
    }

    private mutating func siftUp(from index: Int) {
        var child = index
        let childValue = elements[child]
        var parent = (child - 1) / 2
        while child > 0 && sort(childValue, elements[parent]) {
            elements[child] = elements[parent]
            child = parent
            parent = (child - 1) / 2
        }
        elements[child] = childValue
    }

    private mutating func siftDown(from index: Int) {
        var parent = index
        let count = elements.count
        let parentValue = elements[parent]
        while true {
            let leftChild = 2 * parent + 1
            let rightChild = leftChild + 1
            var candidate = parent
            if leftChild < count && sort(elements[leftChild], elements[candidate]) {
                candidate = leftChild
            }
            if rightChild < count && sort(elements[rightChild], elements[candidate]) {
                candidate = rightChild
            }
            if candidate == parent { break }
            elements[parent] = elements[candidate]
            parent = candidate
        }
        elements[parent] = parentValue
    }
}

// Example Usage
let solution = Solution()
let nums = [3, 2, 1, 5, 6, 4]
let k = 2
print(solution.findKthLargest(nums, k)) // Output: 5



// Swift Solution Using Quickselect (More Efficient)
class Solution {
    func findKthLargest(_ nums: [Int], _ k: Int) -> Int {
        var nums = nums
        let targetIndex = nums.count - k
        return quickselect(&nums, 0, nums.count - 1, targetIndex)
    }

    private func quickselect(_ nums: inout [Int], _ left: Int, _ right: Int, _ k: Int) -> Int {
        let pivotIndex = partition(&nums, left, right)
        if pivotIndex == k {
            return nums[pivotIndex]
        } else if pivotIndex < k {
            return quickselect(&nums, pivotIndex + 1, right, k)
        } else {
            return quickselect(&nums, left, pivotIndex - 1, k)
        }
    }

    private func partition(_ nums: inout [Int], _ left: Int, _ right: Int) -> Int {
        let pivot = nums[right]
        var i = left
        for j in left..<right {
            if nums[j] < pivot {
                nums.swapAt(i, j)
                i += 1
            }
        }
        nums.swapAt(i, right)
        return i
    }
}

// Example Usage
let solution = Solution()
let nums = [3, 2, 1, 5, 6, 4]
let k = 2
print(solution.findKthLargest(nums, k)) // Output: 5

