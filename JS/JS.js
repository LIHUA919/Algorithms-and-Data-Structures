/**
 * @param {number[]} nums
 * @return {number}
 */
// One-dimensional dynamic programming
// Longest Increasing Subsequence

var lengthOfLIS = function (nums) {
  let f = [nums[0]];
  const binarySearch = (x) => {
    let left = 0,
      right = f.length;
    while (left < right) {
      let mid = left + ((right - left) >> 1);
      if (f[mid] < x) left = mid + 1;
      else right = mid;
    }
    return left;
  };
  for (let i = 1; i < nums.length; i++) {
    if (nums[i] > f[f.length - 1]) {
      f.push(nums[i]);
    } else {
      f[binarySearch(nums[i])] = nums[i];
    }
  }
  return f.length;
}

//# Arrays and Strings
//# Remove Duplicates from Sorted Array II

var removeDuplicates = function(nums) {
  const n = nums.length;
  if (n <= 2) {
      return n;
  }
  let slow = 2, fast = 2;
  while (fast < n) {
      if (nums[slow - 2] != nums[fast]) {
          nums[slow] = nums[fast];
          ++slow;
      }
      ++fast;
  }
  return slow;
}


//Rotate Array

const gcd = (x, y) => y ? gcd(y, x % y) : x;

var rotate = function(nums, k) {
    const n = nums.length;
    k = k % n;
    let count = gcd(k, n);
    for (let start = 0; start < count; ++start) {
        let current = start;
        let prev = nums[start];
        do {
            const next = (current + k) % n;
            const temp = nums[next];
            nums[next] = prev;
            prev = temp;
            current = next;
        } while (start !== current);
    }
}


//Best Time to Buy and Sell Stock II
VideoPlaybackQualityvar maxProfit = function(prices) {
  const n = prices.length;
  let dp0 = 0, dp1 = -prices[0];
  for (let i = 1; i < n; ++i) {
      let newDp0 = Math.max(dp0, dp1 + prices[i]);
      let newDp1 = Math.max(dp1, dp0 - prices[i]);
      dp0 = newDp0;
      dp1 = newDp1;
  }
  return dp0;
};



