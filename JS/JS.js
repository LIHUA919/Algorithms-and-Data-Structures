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


var evalRPN = function(tokens) {
  let stack = [];

  for (let token of tokens) {
      if (token === "+" || token === "-" || token === "*" || token === "/") {
          let b = stack.pop();
          let a = stack.pop();
          switch (token) {
              case "+":
                  stack.push(a + b);
                  break;
              case "-":
                  stack.push(a - b);
                  break;
              case "*":
                  stack.push(a * b);
                  break;
              case "/":
                  stack.push(Math.trunc(a / b));
                  break;
          }
      } else {
          stack.push(parseInt(token, 10));
      }
  }

  return stack.pop();
};


var calculate = function(s) {
  let stack = [];
  let currentNumber = 0;
  let result = 0;
  let sign = 1;

  for (let i = 0; i < s.length; i++) {
      let char = s[i];
      
      if (char >= '0' && char <= '9') {
          currentNumber = currentNumber * 10 + (char - '0');
      } else if (char === '+') {
          result += sign * currentNumber;
          currentNumber = 0;
          sign = 1;
      } else if (char === '-') {
          result += sign * currentNumber;
          currentNumber = 0;
          sign = -1;
      } else if (char === '(') {
          stack.push(result);
          stack.push(sign);
          result = 0;
          sign = 1;
      } else if (char === ')') {
          result += sign * currentNumber;
          currentNumber = 0;
          result *= stack.pop(); // pop sign
          result += stack.pop(); // pop previous result
      }
  }
  
  result += sign * currentNumber;
  return result;
};

function ListNode(val, next = null) {
  this.val = val;
  this.next = next;
}

var addTwoNumbers = function(l1, l2) {
  let dummyHead = new ListNode(0);
  let p = l1, q = l2, current = dummyHead;
  let carry = 0;

  while (p !== null || q !== null) {
      let x = (p !== null) ? p.val : 0;
      let y = (q !== null) ? q.val : 0;
      let sum = x + y + carry;
      carry = Math.floor(sum / 10);
      current.next = new ListNode(sum % 10);
      current = current.next;

      if (p !== null) p = p.next;
      if (q !== null) q = q.next;
  }

  if (carry > 0) {
      current.next = new ListNode(carry);
  }

  return dummyHead.next;
};


/**
 * Definition for a binary tree node.
 * function TreeNode(val, left, right) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.left = (left===undefined ? null : left)
 *     this.right = (right===undefined ? null : right)
 * }
 */
/**
 * @param {TreeNode} root
 * @return {TreeNode}
 */
var invertTree = function(root) {
  // Base case: if root is null, return null
  if (root === null) return null;
  
  // Swap the left and right children
  const temp = root.left;
  root.left = root.right;
  root.right = temp;
  
  // Recursively invert left and right subtrees
  invertTree(root.left);
  invertTree(root.right);
  
  return root;
};


/**
 * Definition for a binary tree node.
 * function TreeNode(val, left, right) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.left = (left===undefined ? null : left)
 *     this.right = (right===undefined ? null : right)
 * }
 */
/**
 * @param {TreeNode} root
 * @return {boolean}
 */
var isSymmetric = function(root) {
  // Helper function to check symmetry of two subtrees
  const isMirror = (left, right) => {
      // If both nodes are null, they're symmetric
      if (!left && !right) return true;
      
      // If one node is null and other isn't, not symmetric
      if (!left || !right) return false;
      
      // Check if values are same and subtrees are symmetric
      return (left.val === right.val) && 
             isMirror(left.left, right.right) && 
             isMirror(left.right, right.left);
  };
  
  // Start checking from root's left and right subtrees
  return isMirror(root.left, root.right);
};

/**
 * Definition for a binary tree node.
 * function TreeNode(val, left, right) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.left = (left===undefined ? null : left)
 *     this.right = (right===undefined ? null : right)
 * }
 */
/**
 * @param {number[]} preorder
 * @param {number[]} inorder
 * @return {TreeNode}
 */
var buildTree = function(preorder, inorder) {
  // Base case
  if (!preorder.length || !inorder.length) return null;
  
  // First element in preorder is the root
  const rootVal = preorder[0];
  const root = new TreeNode(rootVal);
  
  // Find root's index in inorder
  const rootIndex = inorder.indexOf(rootVal);
  
  // Recursively build left and right subtrees
  root.left = buildTree(
      preorder.slice(1, rootIndex + 1), 
      inorder.slice(0, rootIndex)
  );
  
  root.right = buildTree(
      preorder.slice(rootIndex + 1), 
      inorder.slice(rootIndex + 1)
  );
  
  return root;
};

/**
 * Definition for a binary tree node.
 * function TreeNode(val, left, right) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.left = (left===undefined ? null : left)
 *     this.right = (right===undefined ? null : right)
 * }
 */
/**
 * @param {number[]} inorder
 * @param {number[]} postorder
 * @return {TreeNode}
 */
var buildTree = function(inorder, postorder) {
  // Base case
  if (!inorder.length || !postorder.length) return null;
  
  // Last element in postorder is the root
  const rootVal = postorder[postorder.length - 1];
  const root = new TreeNode(rootVal);
  
  // Find root's index in inorder
  const rootIndex = inorder.indexOf(rootVal);
  
  // Recursively build left and right subtrees
  root.left = buildTree(
      inorder.slice(0, rootIndex), 
      postorder.slice(0, rootIndex)
  );
  
  root.right = buildTree(
      inorder.slice(rootIndex + 1), 
      postorder.slice(rootIndex, -1)
  );
  
  return root;
};


function connect(root) {
  if (!root) return null;
  
  let queue = [root];
  
  while (queue.length > 0) {
      let size = queue.length;
      let prev = null;
      
      for (let i = 0; i < size; i++) {
          let curr = queue.shift();
          
          if (prev) {
              prev.next = curr;
          }
          prev = curr;
          
          if (curr.left) {
              queue.push(curr.left);
          }
          if (curr.right) {
              queue.push(curr.right);
          }
      }
  }
  
  return root;
}


/**
 * Definition for a binary tree node.
 * function TreeNode(val, left, right) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.left = (left===undefined ? null : left)
 *     this.right = (right===undefined ? null : right)
 * }
 */
/**
 * @param {TreeNode} root
 * @return {void} Do not return anything, modify root in-place instead.
 */
var flatten = function(root) {
  if (!root) return;
  
  let curr = root;
  
  while (curr) {
      if (curr.left) {
          let prev = curr.left;
          while (prev.right) {
              prev = prev.right;
          }
          prev.right = curr.right;
          curr.right = curr.left;
          curr.left = null;
      }
      curr = curr.right;
  }
};


/**
 * Definition for a binary tree node.
 * function TreeNode(val, left, right) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.left = (left===undefined ? null : left)
 *     this.right = (right===undefined ? null : right)
 * }
 */
/**
 * @param {TreeNode} root
 * @param {number} targetSum
 * @return {boolean}
 */
var hasPathSum = function(root, targetSum) {
  if (!root) return false;
  
  targetSum -= root.val;
  if (!root.left && !root.right) return targetSum === 0;
  
  return hasPathSum(root.left, targetSum) || hasPathSum(root.right, targetSum);
};