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
}


/**# Best Time to Buy and Sell Stock
 * enumerate
 * @param {number[]} prices
 * @return {number}
 */
var maxProfit = function(prices) {
    let ans = 0 
    let minPrice = prices[0]
    for (const p of prices) {
        ans = Math.max(ans, p - minPrice)
        minPrice = Math.min(minPrice, p)
    }

    return ans
}


var MinStack = function() {
  this.stack = [];
  this.minStack = [];
};

/** 
* @param {number} val
* @return {void}
*/
MinStack.prototype.push = function(val) {
  this.stack.push(val);
  if (this.minStack.length === 0 || val <= this.minStack[this.minStack.length - 1]) {
      this.minStack.push(val);
  }
};

/**
* @return {void}
*/
MinStack.prototype.pop = function() {
  if (this.stack.pop() === this.minStack[this.minStack.length - 1]) {
      this.minStack.pop();
  }
};

/**
* @return {number}
*/
MinStack.prototype.top = function() {
  return this.stack[this.stack.length - 1];
};

/**
* @return {number}
*/
MinStack.prototype.getMin = function() {
  return this.minStack[this.minStack.length - 1];
};

var simplifyPath = function(path) {
  let stack = [];
  let parts = path.split('/');

  for (let part of parts) {
      if (part === '..') {
          if (stack.length > 0) {
              stack.pop();
          }
      } else if (part !== '' && part !== '.') {
          stack.push(part);
      }
  }

  return '/' + stack.join('/');
};


var isValid = function(s) {
  let stack = [];
  let map = {
      ')': '(',
      '}': '{',
      ']': '['
  };

  for (let char of s) {
      if (char === '(' || char === '{' || char === '[') {
          stack.push(char);
      } else {
          if (stack.length === 0 || stack[stack.length - 1] !== map[char]) {
              return false;
          }
          stack.pop();
      }
  }

  return stack.length === 0;
};



var findMinArrowShots = function(points) {
  if (points.length === 0) return 0;

  // Sort the balloons by their ending point
  points.sort((a, b) => a[1] - b[1]);
  
  let arrows = 1;
  let currentEnd = points[0][1];

  for (let i = 1; i < points.length; i++) {
      // If the current balloon starts after the end of the previous one,
      // we need another arrow
      if (points[i][0] > currentEnd) {
          arrows++;
          currentEnd = points[i][1];
      }
  }

  return arrows;
};


var insert = function(intervals, newInterval) {
  let result = [];
  let i = 0;
  
  // Add all intervals that come before newInterval
  while (i < intervals.length && intervals[i][1] < newInterval[0]) {
      result.push(intervals[i]);
      i++;
  }
  
  // Merge all overlapping intervals with newInterval
  while (i < intervals.length && intervals[i][0] <= newInterval[1]) {
      newInterval[0] = Math.min(newInterval[0], intervals[i][0]);
      newInterval[1] = Math.max(newInterval[1], intervals[i][1]);
      i++;
  }
  result.push(newInterval);
  
  // Add all intervals that come after newInterval
  while (i < intervals.length) {
      result.push(intervals[i]);
      i++;
  }
  
  return result;
};


var twoSum = function(nums, target) {
  let map = new Map();
  for (let i = 0; i < nums.length; i++) {
      let complement = target - nums[i];
      if (map.has(complement)) {
          return [map.get(complement), i];
      }
      map.set(nums[i], i);
  }
};

