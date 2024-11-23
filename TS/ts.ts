// 162
function findPeakElement(nums: number[]): number {
    let left = 0;
    let right = nums.length - 1;

    while (left < right) {
        const mid = Math.floor((left + right) / 2);
        if (nums[mid] > nums[mid + 1]) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }

    return left;
}




function search(nums: number[], target: number): number {
    let left = 0;
    let right = nums.length - 1;

    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        if (nums[mid] === target) {
            return mid;
        }

        if (nums[left] <= nums[mid]) {
            if (nums[left] <= target && target < nums[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        } else {
            if (nums[mid] < target && target <= nums[right]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }

    return -1;
}



function searchRange(nums: number[], target: number): number[] {
    const result = [-1, -1];

    if (nums.length === 0) {
        return result;
    }

    const leftIndex = findLeftIndex(nums, target);
    if (leftIndex === -1) {
        return result;
    }

    result[0] = leftIndex;
    result[1] = findRightIndex(nums, target);

    return result;
}

function findLeftIndex(nums: number[], target: number): number {
    let left = 0;
    let right = nums.length - 1;

    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        if (nums[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    if (left < nums.length && nums[left] === target) {
        return left;
    }

    return -1;
}


function findMin(nums: number[]): number {
    let left = 0;
    let right = nums.length - 1;

    while (left < right) {
        const mid = Math.floor((left + right) / 2);
        if (nums[mid] > nums[right]) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return nums[left];
}

function findMedianSortedArrays(nums1: number[], nums2: number[]): number {
    const m = nums1.length;
    const n = nums2.length;

    if (m > n) {
        return findMedianSortedArrays(nums2, nums1);
    }

    const totalLength = m + n;
    const halfLength = Math.floor(totalLength / 2);

    let left = 0;
    let right = m;

    while (true) {
        const i = Math.floor((left + right) / 2);
        const j = halfLength - i;

        const maxLeftX = i === 0 ? -Infinity : nums1[i - 1];
        const minRightX = i === m ? Infinity : nums1[i];

        const maxLeftY = j === 0 ? -Infinity : nums2[j - 1];
        const minRightY = j === n ? Infinity : nums2[j];

        if (maxLeftX <= minRightY && maxLeftY <= minRightX) {
            if (totalLength % 2 === 0) {
                return (Math.max(maxLeftX, maxLeftY) + Math.min(minRightX, minRightY)) / 2;
            } else {
                return Math.min(minRightX, minRightY);
            }
        } else if (maxLeftX > minRightY) {
            right = i - 1;
        } else {
            left = i + 1;
        }
    }
}



class MaxHeap {
    private heap: number[];

    constructor() {
        this.heap = [];
    }

    insert(val: number) {
        this.heap.push(val);
        this.heapifyUp(this.heap.length - 1);
    }

    extractMax(): number {
        if (this.heap.length === 0) {
            return null;
        }

        if (this.heap.length === 1) {
            return this.heap.pop();
        }

        const max = this.heap[0];
        this.heap[0] = this.heap.pop();
        this.heapifyDown(0);

        return max;
    }

    private heapifyUp(index: number) {
        if (index === 0) {
            return;
        }

        const parentIndex = Math.floor((index - 1) / 2);
        if (this.heap[parentIndex] < this.heap[index]) {
            [this.heap[parentIndex], this.heap[index]] = [this.heap[index], this.heap[parentIndex]];
            this.heapifyUp(parentIndex);
        }
    }

    private heapifyDown(index: number) {
        const leftChildIndex = 2 * index + 1;
        const rightChildIndex = 2 * index + 2;
        let largestIndex = index;

        if (leftChildIndex < this.heap.length && this.heap[leftChildIndex] > this.heap[largestIndex]) {
            largestIndex = leftChildIndex;
        }

        if (rightChildIndex < this.heap.length && this.heap[rightChildIndex] > this.heap[largestIndex]) {
            largestIndex = rightChildIndex;
        }

        if (largestIndex !== index) {
            [this.heap[largestIndex], this.heap[index]] = [this.heap[index], this.heap[largestIndex]];
            this.heapifyDown(largestIndex);
        }
    }
}

function findKthLargest(nums: number[], k: number): number {
    const maxHeap = new MaxHeap();

    for (const num of nums) {
        maxHeap.insert(num);
    }

    for (let i = 0; i < k - 1; i++) {
        maxHeap.extractMax();
    }

    return maxHeap.extractMax();
}