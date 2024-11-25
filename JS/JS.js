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



/**
 * Definition for a binary tree node.
 * function TreeNode(val, left, right) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.left = (left===undefined ? null : left)
 *     this.right = (right===undefined ? null : right)
 * }
 */
class BSTIterator {
  constructor(root) {
      // Initialize stack for storing nodes
      this.stack = [];
      
      // Helper function to push all left nodes onto stack
      const pushLeft = (node) => {
          while (node) {
              this.stack.push(node);
              node = node.left;
          }
      }
      
      // Initialize stack with leftmost path
      pushLeft(root);
  }
  
  hasNext() {
      // If stack is not empty, we have more nodes to process
      return this.stack.length > 0;
  }
  
  next() {
      // Get next node from top of stack
      const current = this.stack.pop();
      
      // If current node has right child,
      // push all left nodes of right subtree onto stack
      if (current.right) {
          let temp = current.right;
          while (temp) {
              this.stack.push(temp);
              temp = temp.left;
          }
      }
      
      return current.val;
  }
}

/** 
* Your BSTIterator object will be instantiated and called as such:
* var obj = new BSTIterator(root)
* var param_1 = obj.hasNext()
* var param_2 = obj.next()
*/

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
 * @return {number}
 */
const countNodes = function(root) {
  if (!root) return 0;
  
  // Get height of left and right edges
  const getLeftHeight = (node) => {
      let height = 0;
      while (node) {
          height++;
          node = node.left;
      }
      return height;
  };
  
  const getRightHeight = (node) => {
      let height = 0;
      while (node) {
          height++;
          node = node.right;
      }
      return height;
  };
  
  // Get left and right heights
  const leftHeight = getLeftHeight(root);
  const rightHeight = getRightHeight(root);
  
  // If heights are equal, tree is perfect binary tree
  if (leftHeight === rightHeight) {
      return Math.pow(2, leftHeight) - 1;
  }
  
  // Otherwise, recursively count nodes
  return 1 + countNodes(root.left) + countNodes(root.right);
};

/**
 * Definition for a binary tree node.
 * function TreeNode(val) {
 *     this.val = val;
 *     this.left = this.right = null;
 * }
 */
/**
 * @param {TreeNode} root
 * @param {TreeNode} p
 * @param {TreeNode} q
 * @return {TreeNode}
 */
const lowestCommonAncestor = function(root, p, q) {
  // Base cases
  if (!root || root === p || root === q) {
      return root;
  }
  
  // Look for p and q in left and right subtrees
  const leftResult = lowestCommonAncestor(root.left, p, q);
  const rightResult = lowestCommonAncestor(root.right, p, q);
  
  // If we found both p and q in different subtrees,
  // current node is the LCA
  if (leftResult && rightResult) {
      return root;
  }
  
  // If we found only one node, return that node
  // (it could be the LCA if the other node is its descendant)
  return leftResult || rightResult;
};



/**
 * @param {TreeNode} root
 * @return {number[]}
 */
const rightSideView = function(root) {
  const result = [];
  
  function dfs(node, level) {
      if (!node) return;
      
      // If this is the first node we've seen at this level
      // it must be the rightmost node when traversing right first
      if (level === result.length) {
          result.push(node.val);
      }
      
      // Visit right child first to ensure we see rightmost nodes first
      if (node.right) dfs(node.right, level + 1);
      if (node.left) dfs(node.left, level + 1);
  }
  
  dfs(root, 0);
  return result;
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
 * @return {number[]}
 */
var averageOfLevels = function(root) {
  if (!root) return [];
  
  const result = [];
  const queue = [root];
  
  while (queue.length) {
      const levelSize = queue.length;
      let levelSum = 0;
      
      for (let i = 0; i < levelSize; i++) {
          const node = queue.shift();
          levelSum += node.val;
          
          if (node.left) queue.push(node.left);
          if (node.right) queue.push(node.right);
      }
      
      result.push(levelSum / levelSize);
  }
  
  return result;
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
 * @return {number[][]}
 */
var levelOrder = function(root) {
  if (!root) return [];
  
  const result = [];
  const queue = [root];
  
  while (queue.length) {
      const levelSize = queue.length;
      const currentLevel = [];
      
      for (let i = 0; i < levelSize; i++) {
          const node = queue.shift();
          currentLevel.push(node.val);
          
          if (node.left) queue.push(node.left);
          if (node.right) queue.push(node.right);
      }
      
      result.push(currentLevel);
  }
  
  return result;
};

/**
 * Definition for a binary tree node.
 * function TreeNode(val, left, right) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.left = (left===undefined ? null : left)
 *     this.right = (right===undefined ? null : right)
 * }
 */
function zigzagLevelOrder(root) {
  if (!root) return [];
  
  const result = [];
  const queue = [root];
  let isLeftToRight = true;
  
  while (queue.length) {
      const levelSize = queue.length;
      const currentLevel = [];
      
      for (let i = 0; i < levelSize; i++) {
          const node = queue.shift();
          
          // Add values based on current direction
          if (isLeftToRight) {
              currentLevel.push(node.val);
          } else {
              currentLevel.unshift(node.val);
          }
          
          // Add children to queue
          if (node.left) queue.push(node.left);
          if (node.right) queue.push(node.right);
      }
      
      result.push(currentLevel);
      isLeftToRight = !isLeftToRight;
  }
  
  return result;
}



/**
 * @param {character[][]} board
 * @return {void} Do not return anything, modify board in-place instead.
 */
var solve = function(board) {
    if (!board || !board.length) return;
    
    const m = board.length;
    const n = board[0].length;
    
    // Mark connected boundary O's as safe (change to 'S')
    // Check first and last row
    for (let j = 0; j < n; j++) {
        dfs(board, 0, j);        // First row
        dfs(board, m-1, j);      // Last row
    }
    
    // Check first and last column
    for (let i = 0; i < m; i++) {
        dfs(board, i, 0);        // First column
        dfs(board, i, n-1);      // Last column
    }
    
    // Process the board: O -> X (captured) and S -> O (safe)
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            if (board[i][j] === 'O') {
                board[i][j] = 'X';  // Capture surrounded O's
            } else if (board[i][j] === 'S') {
                board[i][j] = 'O';  // Restore safe O's
            }
        }
    }
};

// DFS to mark connected O's as safe
function dfs(board, i, j) {
    // Check bounds and if current cell is 'O'
    if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || board[i][j] !== 'O') {
        return;
    }
    
    // Mark as safe
    board[i][j] = 'S';
    
    // Check all 4 directions
    dfs(board, i+1, j);  // Down
    dfs(board, i-1, j);  // Up
    dfs(board, i, j+1);  // Right
    dfs(board, i, j-1);  // Left
}













/**
 * // Definition for a Node.
 * function Node(val, neighbors) {
 *    this.val = val === undefined ? 0 : val;
 *    this.neighbors = neighbors === undefined ? [] : neighbors;
 * };
 */

/**
 * @param {Node} node
 * @return {Node}
 */
var cloneGraph = function(node) {
    // Handle edge case of empty graph
    if (!node) return null;
    
    // Map to store the cloned nodes
    // Key: original node value, Value: cloned node
    const visited = new Map();
    
    function dfs(node) {
        // If we've already cloned this node, return the clone
        if (visited.has(node.val)) {
            return visited.get(node.val);
        }
        
        // Create a new node with the same value
        const clone = new Node(node.val);
        
        // Add it to visited map before DFS to handle cycles
        visited.set(node.val, clone);
        
        // Clone all neighbors
        for (let neighbor of node.neighbors) {
            clone.neighbors.push(dfs(neighbor));
        }
        
        return clone;
    }
    
    return dfs(node);
};











/**
 * @param {number} numCourses
 * @param {number[][]} prerequisites
 * @return {boolean}
 */
var canFinish = function(numCourses, prerequisites) {
    // Create adjacency list to represent the graph
    const graph = Array(numCourses).fill().map(() => []);
    // Track in-degree for each node (course)
    const inDegree = Array(numCourses).fill(0);
    
    // Build the graph and count in-degrees
    for (const [course, prereq] of prerequisites) {
        graph[prereq].push(course);
        inDegree[course]++;
    }
    
    // Queue for courses that have no prerequisites (in-degree = 0)
    const queue = [];
    
    // Add all courses with no prerequisites to queue
    for (let i = 0; i < numCourses; i++) {
        if (inDegree[i] === 0) {
            queue.push(i);
        }
    }
    
    // Counter for completed courses
    let completed = 0;
    
    // Process the queue (topological sort)
    while (queue.length > 0) {
        const current = queue.shift();
        completed++;
        
        // For each course that depends on current course
        for (const nextCourse of graph[current]) {
            inDegree[nextCourse]--;
            
            // If all prerequisites are completed for this course
            if (inDegree[nextCourse] === 0) {
                queue.push(nextCourse);
            }
        }
    }
    
    // If we completed all courses, there's no cycle
    return completed === numCourses;
};









// 秒出模型：cerebras
var snakesAndLadders = function(board) {
    const n = board.length;
    const visited = new Set();
    const queue = [[1, 0]];
    visited.add(1);

    while (queue.length) {
        const [curr, rolls] = queue.shift();
        if (curr === n * n) return rolls;

        for (let i = 1; i <= 6; i++) {
            const next = curr + i;
            if (next > n * n) continue;

            const [row, col] = getCoordinates(next, n);
            const value = board[row][col];
            let destination = next;
            if (value !== -1) destination = value;

            if (!visited.has(destination)) {
                queue.push([destination, rolls + 1]);
                visited.add(destination);
            }
        }
    }

    return -1;

    function getCoordinates(num, n) {
        const row = Math.floor((num - 1) / n);
        const col = (num - 1) % n;
        if (row % 2 === 0) return [n - row - 1, col];
        else return [n - row - 1, n - col - 1];
    }
};










var minMutation = function(startGene, endGene, bank) {
    const set = new Set(bank);
    const queue = [[startGene, 0]];
    const visited = new Set([startGene]);

    while (queue.length) {
        const [gene, steps] = queue.shift();
        if (gene === endGene) return steps;

        for (let i = 0; i < 8; i++) {
            for (const char of 'ACGT') {
                if (gene[i] === char) continue;
                const nextGene = gene.slice(0, i) + char + gene.slice(i + 1);
                if (!set.has(nextGene) || visited.has(nextGene)) continue;
                queue.push([nextGene, steps + 1]);
                visited.add(nextGene);
            }
        }
    }

    return -1;
};



var combine = function(n, k) {
    const result = [];

    function backtrack(start, path) {
        if (path.length === k) {
            result.push([...path]);
            return;
        }

        for (let i = start; i <= n; i++) {
            path.push(i);
            backtrack(i + 1, path);
            path.pop();
        }
    }

    backtrack(1, []);
    return result;
};



var maximalSquare = function(matrix) {
    if (matrix.length === 0) return 0;

    let rows = matrix.length, cols = matrix[0].length;
    let dp = Array(rows).fill(0).map(() => Array(cols).fill(0));
    let maxSquareLength = 0;

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            if (matrix[i][j] === '1') {
                if (i === 0 || j === 0) {
                    dp[i][j] = 1; // For first row or column
                } else {
                    dp[i][j] = Math.min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1;
                }
                maxSquareLength = Math.max(maxSquareLength, dp[i][j]);
            }
        }
    }

    return maxSquareLength * maxSquareLength;
};


var maxProfit = function(k, prices) {
    let n = prices.length;
    if (n === 0 || k === 0) return 0;

    // If k is greater than or equal to n/2, treat it as unlimited transactions
    if (k >= Math.floor(n / 2)) {
        let profit = 0;
        for (let i = 1; i < n; i++) {
            if (prices[i] > prices[i - 1]) {
                profit += prices[i] - prices[i - 1];
            }
        }
        return profit;
    }

    // Initialize DP arrays
    let dpPrev = new Array(n).fill(0);
    let dpCurr = new Array(n).fill(0);

    for (let i = 1; i <= k; i++) {
        let maxDiff = -prices[0];
        for (let j = 1; j < n; j++) {
            dpCurr[j] = Math.max(dpCurr[j - 1], prices[j] + maxDiff);
            maxDiff = Math.max(maxDiff, dpPrev[j] - prices[j]);
        }
        dpPrev = dpCurr.slice(); // Update the previous transaction's state
    }

    return dpPrev[n - 1];
};










var maxProfit = function(prices) {
    if (prices.length < 2) return 0;

    let n = prices.length;

    // Arrays to store max profit up to day i and from day i
    let leftProfit = new Array(n).fill(0);
    let rightProfit = new Array(n).fill(0);

    // Calculate leftProfit: max profit up to each day
    let minPrice = prices[0];
    for (let i = 1; i < n; i++) {
        minPrice = Math.min(minPrice, prices[i]);
        leftProfit[i] = Math.max(leftProfit[i - 1], prices[i] - minPrice);
    }

    // Calculate rightProfit: max profit from each day to the end
    let maxPrice = prices[n - 1];
    for (let i = n - 2; i >= 0; i--) {
        maxPrice = Math.max(maxPrice, prices[i]);
        rightProfit[i] = Math.max(rightProfit[i + 1], maxPrice - prices[i]);
    }

    // Combine the results
    let maxProfit = 0;
    for (let i = 0; i < n; i++) {
        maxProfit = Math.max(maxProfit, leftProfit[i] + rightProfit[i]);
    }

    return maxProfit;
};


# Creating a single file with the implementation of the solution and examples for the problem.

code = """
/**
 * Function to calculate the minimum number of operations required to convert word1 to word2.
 * @param {string} word1 - The first string.
 * @param {string} word2 - The second string.
 * @return {number} - The minimum number of operations required.
 */
var minDistance = function(word1, word2) {
    const m = word1.length;
    const n = word2.length;

    // Create a DP table
    const dp = Array(m + 1).fill(0).map(() => Array(n + 1).fill(0));

    // Fill the base cases
    for (let i = 0; i <= m; i++) dp[i][0] = i;
    for (let j = 0; j <= n; j++) dp[0][j] = j;

    // Fill the DP table
    for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
            if (word1[i - 1] === word2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
            }
        }
    }

    return dp[m][n];
};








// Example Usage:
const word1 = "horse";
const word2 = "ros";
console.log("Minimum Operations to Convert:", minDistance(word1, word2)); // Output: 3

const word3 = "intention";
const word4 = "execution";
console.log("Minimum Operations to Convert:", minDistance(word3, word4)); // Output: 5

const word5 = "abc";
const word6 = "yabd";
console.log("Minimum Operations to Convert:", minDistance(word5, word6)); // Output: 2
//"""

//# Save the code to a JavaScript file
file_path = "/mnt/data/MinDistanceOperations.js"
with open(file_path, "w") as file:
    file.write(code)

file_path










// JavaScript solution to determine if s3 is formed by an interleaving of s1 and s2
function isInterleave(s1, s2, s3) {
    if (s1.length + s2.length !== s3.length) {
        return false;
    }

    const dp = Array.from({ length: s1.length + 1 }, () => Array(s2.length + 1).fill(false));
    dp[0][0] = true;

    // Fill in the first row
    for (let j = 1; j <= s2.length; j++) {
        dp[0][j] = dp[0][j - 1] && s2[j - 1] === s3[j - 1];
    }

    // Fill in the first column
    for (let i = 1; i <= s1.length; i++) {
        dp[i][0] = dp[i - 1][0] && s1[i - 1] === s3[i - 1];
    }

    // Fill in the rest of the DP table
    for (let i = 1; i <= s1.length; i++) {
        for (let j = 1; j <= s2.length; j++) {
            dp[i][j] = (dp[i - 1][j] && s1[i - 1] === s3[i + j - 1]) ||
                       (dp[i][j - 1] && s2[j - 1] === s3[i + j - 1]);
        }
    }

    return dp[s1.length][s2.length];
}

// Example usage
const s1 = "aab";
const s2 = "axy";
const s3 = "aaxaaby";
console.log(isInterleave(s1, s2, s3));  // Output: true


// JavaScript solution to find the longest palindromic substring in a given string
function longestPalindrome(s) {
    if (s.length < 1) return "";

    let start = 0, end = 0;

    const expandAroundCenter = (s, left, right) => {
        while (left >= 0 && right < s.length && s[left] === s[right]) {
            left--;
            right++;
        }
        return right - left - 1;
    };

    for (let i = 0; i < s.length; i++) {
        let len1 = expandAroundCenter(s, i, i);
        let len2 = expandAroundCenter(s, i, i + 1);
        let len = Math.max(len1, len2);
        if (len > end - start) {
            start = i - Math.floor((len - 1) / 2);
            end = i + Math.floor(len / 2);
        }
    }

    return s.substring(start, end + 1);
}

// Example usage
const s = "babad";
console.log(longestPalindrome(s));  // Output: "bab" or "aba"













/**
 * @param {number[][]} obstacleGrid
 * @return {number}
 */
var uniquePathsWithObstacles = function(obstacleGrid) {
    const m = obstacleGrid.length;
    const n = obstacleGrid[0].length;
    
    // If the start or end is an obstacle, return 0
    if (obstacleGrid[0][0] === 1 || obstacleGrid[m - 1][n - 1] === 1) {
        return 0;
    }
    




    // Create a 2D DP array to store the number of ways to reach each cell
    const dp = Array.from({ length: m }, () => Array(n).fill(0));
    
    // Start position
    dp[0][0] = 1;



    
    // Fill the DP table
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            if (obstacleGrid[i][j] === 1) {
                dp[i][j] = 0; // Obstacle cell
            } else {
                if (i > 0) {
                    dp[i][j] += dp[i - 1][j]; // From top
                }
                if (j > 0) {
                    dp[i][j] += dp[i][j - 1]; // From left
                }
            }
        }
    }


    
    // The answer is in the bottom-right corner
    return dp[m - 1][n - 1];
};



// Example usage
const grid = [
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
];
console.log(uniquePathsWithObstacles(grid));  // Output: 2



/**
 * @param {number[][]} matrix
 * @return {number[]}
 */
var spiralOrder = function(matrix) {
    if (matrix.length === 0) return [];
    
    const result = [];
    let top = 0;
    let bottom = matrix.length - 1;
    let left = 0;
    let right = matrix[0].length - 1;
    
    while (top <= bottom && left <= right) {
        // Traverse from left to right along the top row
        for (let i = left; i <= right; i++) {
            result.push(matrix[top][i]);
        }
        top++;
        
        // Traverse from top to bottom along the right column
        for (let i = top; i <= bottom; i++) {
            result.push(matrix[i][right]);
        }
        right--;
        
        if (top <= bottom) {
            // Traverse from right to left along the bottom row
            for (let i = right; i >= left; i--) {
                result.push(matrix[bottom][i]);
            }
            bottom--;
        }
        
        if (left <= right) {
            // Traverse from bottom to top along the left column
            for (let i = bottom; i >= top; i--) {
                result.push(matrix[i][left]);
            }
            left++;
        }
    }
    
    return result;
};

// Example usage
const matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
];
console.log(spiralOrder(matrix));  // Output: [1, 2, 3, 6, 9, 8, 7, 4, 5]
