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








/**
 * @param {number[][]} board
 * @return {void} Do not return anything, modify board in-place instead.
 */
var gameOfLife = function(board) {
    const m = board.length;
    const n = board[0].length;
    
    // Directions representing the 8 neighbors of a cell
    const directions = [
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1],         [0, 1],
        [1, -1], [1, 0], [1, 1]
    ];
    
    // Iterate through each cell in the board
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            let liveNeighbors = 0;
            
            // Count the live neighbors
            for (const [dx, dy] of directions) {
                const x = i + dx;
                const y = j + dy;
                if (x >= 0 && x < m && y >= 0 && y < n && Math.abs(board[x][y]) === 1) {
                    liveNeighbors++;
                }
            }
            
            // Apply the rules of the Game of Life
            if (board[i][j] === 1 && (liveNeighbors < 2 || liveNeighbors > 3)) {
                board[i][j] = -1; // Mark as a cell that was live but is now dead
            }
            if (board[i][j] === 0 && liveNeighbors === 3) {
                board[i][j] = 2; // Mark as a cell that was dead but is now live
            }
        }
    }
    
    // Update the board to the next state
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            if (board[i][j] === -1) {
                board[i][j] = 0;
            } else if (board[i][j] === 2) {
                board[i][j] = 1;
            }
        }
    }
};

// Example usage
const board = [
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1],
    [0, 0, 0]
];
gameOfLife(board);
console.log(board);  // Output: [[0, 0, 0], [1, 0, 1], [0, 1, 1], [0, 1, 0]];




const isPrime = (x) => {
    for (let i = 2; i * i <= x; ++i) {
        if (x % i == 0) {
            return false;
        }
    }
    return true;
}

var countPrimes = function(n) {
    let ans = 0;
    for (let i = 2; i < n; ++i) {
        ans += isPrime(i);
    }
    return ans;
};

var findCircleNum = function(isConnected) {
    const cities = isConnected.length;
    const visited = new Set();
    let provinces = 0;
    for (let i = 0; i < cities; i++) {
        if (!visited.has(i)) {
            dfs(isConnected, visited, cities, i);
            provinces++;
        }
    }
    return provinces;
};

const dfs = (isConnected, visited, cities, i) => {
    for (let j = 0; j < cities; j++) {
        if (isConnected[i][j] == 1 && !visited.has(j)) {
            visited.add(j);
            dfs(isConnected, visited, cities, j);
        }
    }
};

var allPathsSourceTarget = function(graph) {
    const stack = [], ans = [];

    const dfs = (graph, x, n) => {
        if (x === n) {
            ans.push(stack.slice());
            return;
        }
        for (const y of graph[x]) {
            stack.push(y);
            dfs(graph, y, n);
            stack.pop();
        }
    }

    stack.push(0);
    dfs(graph, 0, graph.length - 1);
    return ans;
};


var findCheapestPrice = function(n, flights, src, dst, k) {
    const INF = 10000 * 101 + 1;
    const f = new Array(k + 2).fill(0).map(() => new Array(n).fill(INF));
    f[0][src] = 0;
    for (let t = 1; t <= k + 1; ++t) {
        for (const flight of flights) {
            const j = flight[0], i = flight[1], cost = flight[2];
            f[t][i] = Math.min(f[t][i], f[t - 1][j] + cost);
        }
    }
    let ans = INF;
    for (let t = 1; t <= k + 1; ++t) {
        ans = Math.min(ans, f[t][dst]);
    }
    return ans == INF ? -1 : ans;
};


var reversePairs = function(nums) {
    if (nums.length === 0) {
        return 0;
    }
    return reversePairsRecursive(nums, 0, nums.length - 1);
};

const reversePairsRecursive = (nums, left, right) => {
    if (left === right) {
        return 0;
    } else {
        const mid = Math.floor((left + right) / 2);
        const n1 = reversePairsRecursive(nums, left, mid);
        const n2 = reversePairsRecursive(nums, mid + 1, right);
        let ret = n1 + n2;

        let i = left;
        let j = mid + 1;
        while (i <= mid) {
            while (j <= right && nums[i] > 2 * nums[j]) {
                j++;
            }
            ret += j - mid - 1;
            i++;
        }

        const sorted = new Array(right - left + 1);
        let p1 = left, p2 = mid + 1;
        let p = 0;
        while (p1 <= mid || p2 <= right) {
            if (p1 > mid) {
                sorted[p++] = nums[p2++];
            } else if (p2 > right) {
                sorted[p++] = nums[p1++];
            } else {
                if (nums[p1] < nums[p2]) {
                    sorted[p++] = nums[p1++];
                } else {
                    sorted[p++] = nums[p2++];
                }
            }
        }
        for (let k = 0; k < sorted.length; k++) {
            nums[left + k] = sorted[k];
        }
        return ret;
    }
}

var canCross = function(stones) {
    const n = stones.length;
    rec = new Array(n).fill(0).map(() => new Map());

    const dfs = (stones, i, lastDis) => {
        if (i === stones.length - 1) {
            return true;
        }
        if (rec[i].has(lastDis)) {
            return rec[i].get(lastDis);
        }
        for (let curDis = lastDis - 1; curDis <= lastDis + 1; curDis++) {
            if (curDis > 0) {
                const j = lower_bound(stones, curDis + stones[i]);
                if (j !== stones.length && stones[j] === curDis + stones[i] && dfs(stones, j, curDis)) {
                    rec[i].set(lastDis, true);
                    return rec[i].get(lastDis);
                }
            }
        }
        rec[i].set(lastDis, false);
        return rec[i].get(lastDis);
    }

    return dfs(stones, 0, 0);
};

function lower_bound(nums, target) {
    let lo = 0, hi = nums.length;

    while (lo < hi) {
        const mid = lo + Math.floor((hi - lo) / 2);

        if (nums[mid] >= target) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}



var canPartition = function(nums) {
    const n = nums.length;
    if (n < 2) {
        return false;
    }
    let sum = 0, maxNum = 0;
    for (const num of nums) {
        sum += num;
        maxNum = maxNum > num ? maxNum : num;
    }
    if (sum & 1) {
        return false;
    }
    const target = Math.floor(sum / 2);
    if (maxNum > target) {
        return false;
    }
    const dp = new Array(n).fill(0).map(() => new Array(target + 1, false));
    for (let i = 0; i < n; i++) {
        dp[i][0] = true;
    }
    dp[0][nums[0]] = true;
    for (let i = 1; i < n; i++) {
        const num = nums[i];
        for (let j = 1; j <= target; j++) {
            if (j >= num) {
                dp[i][j] = dp[i - 1][j] | dp[i - 1][j - num];
            } else {
                dp[i][j] = dp[i - 1][j];
            }
        }
    }
    return dp[n - 1][target];
};



const isPrime = (x) => {
    for (let i = 2; i * i <= x; ++i) {
        if (x % i == 0) {
            return false;
        }
    }
    return true;
}

var countPrimes = function(n) {
    let ans = 0;
    for (let i = 2; i < n; ++i) {
        ans += isPrime(i);
    }
    return ans;
};




var removeDuplicateLetters = function(s) {
    const vis = new Array(26).fill(0);
    const num = _.countBy(s);
    
    const sb = new Array();
    for (let i = 0; i < s.length; i++) {
        const ch = s[i];
        if (!vis[ch.charCodeAt() - 'a'.charCodeAt()]) {
            while (sb.length > 0 && sb[sb.length - 1] > ch) {
                if (num[sb[sb.length - 1]] > 0) {
                    vis[sb[sb.length - 1].charCodeAt() - 'a'.charCodeAt()] = 0;
                    sb.pop();
                } else {
                    break;
                }
            }
            vis[ch.charCodeAt() - 'a'.charCodeAt()] = 1;
            sb.push(ch);
        }
        num[ch]--;
    }
    return sb.join('');
};



/**
 * @param {string} stamp
 * @param {string} target
 * @return {number[]}
 */


var movesToStamp = function(stamp, target) {
    const ans = [];
    let change = 0;
    const m = stamp.length;
    target = target.split('');
    const n = target.length;
    
    while (true) {
      let someMatch = false; // 在 target 中是否某些位置开始可以匹配
  
      for (let i = 0; i < n; i++) {
        let allMark = true; // 是否全是 ?
        let currMatch = true; // 是否 i 开始，stamp 可以全部匹配(字符相同或者是 ? 都表示匹配)
  
        for (let j = 0; j < m; j++) {
          if (target[i + j] !== '?') {
            allMark = false;
          }
          if (target[i + j] === '?' || target[i + j] === stamp[j]) {
            continue;
          } else {
            currMatch = false;
            break;
          }
        }
  
        // 当前 i 位置匹配 stamp，并且匹配的位置还有的不是 ?，把对应位置变成 ?，并记录改变的数量
        if (!allMark && currMatch) {
          ans.push(i);
          for (let j = 0; j < m; j++) {
            if (target[i + j] !== '?') {
              target[i + j] = '?';
              change++;
            }
          }
          someMatch = true;
        }
      }
  
      // target 所有位置尝试过了，没有能匹配的情况了
      if (!someMatch) {
        return [];
      }
      // 字符变成 ? 的数量等于 target.length，完全匹配，返回结果
      if (change === n) {
        ans.reverse();
        return ans;
      }
    }
  };
  





  function nextPermutation(nums) {
    let i = nums.length - 2;                   // 向左遍历，i从倒数第二开始是为了nums[i+1]要存在
    while (i >= 0 && nums[i] >= nums[i + 1]) { // 寻找第一个小于右邻居的数
        i--;
    }
    if (i >= 0) {                             // 这个数在数组中存在，从它身后挑一个数，和它换
        let j = nums.length - 1;                // 从最后一项，向左遍历
        while (j >= 0 && nums[j] <= nums[i]) {  // 寻找第一个大于 nums[i] 的数
            j--;
        }
        [nums[i], nums[j]] = [nums[j], nums[i]]; // 两数交换，实现变大
    }
    // 如果 i = -1，说明是递减排列，如 3 2 1，没有下一排列，直接翻转为最小排列：1 2 3
    let l = i + 1;           
    let r = nums.length - 1;
    while (l < r) {                            // i 右边的数进行翻转，使得变大的幅度小一些
        [nums[l], nums[r]] = [nums[r], nums[l]];
        l++;
        r--;
    }
}

var largestComponentSize = function(nums) {
    const m = _.max(nums);;
    const uf = new UnionFind(m + 1);
    for (const num of nums) {
        for (let i = 2; i * i <= num; i++) {
            if (num % i === 0) {
                uf.union(num, i);
                uf.union(num, Math.floor(num / i));
            }
        }
    }
    const counts = new Array(m + 1).fill(0);
    let ans = 0;
    for (let num of nums) {
        const root = uf.find(num);
        counts[root]++;
        ans = Math.max(ans, counts[root]);
    }
    return ans;
};

class UnionFind {
    constructor(n) {
        this.parent = new Array(n).fill(0).map((_, i) => i);
        this.rank = new Array(n).fill(0);
    }

    union(x, y) {
        let rootx = this.find(x);
        let rooty = this.find(y);
        if (rootx !== rooty) {
            if (this.rank[rootx] > this.rank[rooty]) {
                this.parent[rooty] = rootx;
            } else if (this.rank[rootx] < this.rank[rooty]) {
                this.parent[rootx] = rooty;
            } else {
                this.parent[rooty] = rootx;
                this.rank[rootx]++;
            }
        }
    }

    find(x) {
        if (this.parent[x] !== x) {
            this.parent[x] = this.find(this.parent[x]);
        }
        return this.parent[x];
    }
}

const MOD = 1e9 + 7;
const MAXN = 1e4 + 14;
const MAXM = 14;

var waysToFillArray = function(queries) {
    const comb = new Array(MAXN).fill(null).map(() => new Array(MAXM).fill(0));
    const ans = [];

    comb[0][0] = 1;
    for (let i = 1; i < MAXN; i++) {
        comb[i][0] = 1;
        for (let j = 1; j <= i && j < MAXM; j++) {
            comb[i][j] = (comb[i - 1][j - 1] + comb[i - 1][j]) % MOD;
        }
    }

    for (const q of queries) {
        let [n, k] = q;
        let tot = BigInt(1);
        for (let i = 2; i * i <= k; i++) {
            if (k % i === 0) {
                let cnt = 0;
                while (k % i === 0) {
                    k /= i;
                    cnt++;
                }
                tot = (tot * BigInt(comb[n + cnt - 1][cnt])) % BigInt(MOD);
            }
        }
        // k 自身为质数
        if (k > 1) {
            tot = (tot * BigInt(n)) % BigInt(MOD);
        }
        ans.push(Number(tot));
    }
    return ans;
};



var kthFactor = function(n, k) {
    const arr = [];
    for(let i = 1; i <= n/2; i++){
        if(n % i == 0) arr.push(i);
    }
    arr.push(n);
    return arr[k-1] || -1;
};

  

/**
 * @param {number[]} nums
 * @return {number}
 */
var maxCoins = function (nums) {
    let n = nums.length;
    // 添加两侧的虚拟气球
    let points = new Array(n + 2);
    points[0] = points[n + 1] = 1;
    for (let i = 1; i <= n; i++) {
      points[i] = nums[i - 1];
    }
    // base case 已经都被初始化为 0
    let dp = new Array(n + 2).fill(0).map(() => new Array(n + 2).fill(0));
    // 开始状态转移
    // i 应该从下往上
    for (let i = n; i >= 0; i--) {
      // j 应该从左往右
      for (let j = i + 1; j < n + 2; j++) {
        // 最后戳破的气球是哪个？
        for (let k = i + 1; k < j; k++) {
          // 择优做选择
          dp[i][j] = Math.max(
            dp[i][j],
            dp[i][k] + dp[k][j] + points[i] * points[j] * points[k]
          );
        }
      }
    }
    return dp[0][n + 1];
  };
  

  



/**
 * @param {character[][]} matrix
 * @return {number}
 */
var maximalRectangle = function(matrix) {
    
};

var maximalRectangle = function(matrix) {
    const m = matrix.length;
    if (m === 0) {
        return 0;
    }
    const n = matrix[0].length;
    const left = new Array(m).fill(0).map(() => new Array(n).fill(0));

    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            if (matrix[i][j] === '1') {
                left[i][j] = (j === 0 ? 0 : left[i][j - 1]) + 1;
            }
        }
    }

    let ret = 0;
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            if (matrix[i][j] === '0') {
                continue;
            }
            let width = left[i][j];
            let area = width;
            for (let k = i - 1; k >= 0; k--) {
                width = Math.min(width, left[k][j]);
                area = Math.max(area, (i - k + 1) * width);
            }
            ret = Math.max(ret, area);
        }
    }
    return ret;
};


/**
 * @param {number[]} nums
 * @return {number}
 */
var lengthOfLIS = function(nums) {

    // 每堆的堆顶
    const top = [];
    // 牌堆数初始化为0
    let piles = 0;
    for (let i = 0; i < nums.length; i++) {
      // 要处理的扑克牌
      let poker = nums[i];
      // 左堆和最右堆进行二分搜索，因为堆顶是有序排的，最终找到该牌要插入的堆
      let left = 0,
        right = piles;
      //搜索区间是左闭右开
      while (left < right) {
        let mid = left + ((right - left) >> 1);
        if (top[mid] > poker) {
          right = mid;
        } else if (top[mid] < poker) {
          left = mid + 1;
        } else {
          right = mid;
        }
      }
  
      //  没找到合适的牌堆，新建一堆
      if (left == piles) piles++;
      // 把这张牌放到堆顶
      top[left] = poker;
    }
    return piles;
  };

  var minValidStrings = function(words, target) {
    const prefixFunction = (word, target) => {
        const s = word + '#' + target;
        const n = s.length;
        const pi = new Array(n).fill(0);
        for (let i = 1; i < n; i++) {
            let j = pi[i - 1];
            while (j > 0 && s[i] !== s[j]) {
                j = pi[j - 1];
            }
            if (s[i] === s[j]) {
                j++;
            }
            pi[i] = j;
        }
        return pi;
    };

    const n = target.length;
    const back = new Array(n).fill(0);
    for (const word of words) {
        const pi = prefixFunction(word, target);
        const m = word.length;
        for (let i = 0; i < n; i++) {
            back[i] = Math.max(back[i], pi[m + 1 + i]);
        }
    }

    const dp = new Array(n + 1).fill(0);
    for (let i = 1; i <= n; i++) {
        dp[i] = 1e9;
    }
    for (let i = 0; i < n; i++) {
        dp[i + 1] = dp[i + 1 - back[i]] + 1;
        if (dp[i + 1] > n) {
            return -1;
        }
    }
    return dp[n];
};



var maximumDetonation = function(bombs) {
    const n = bombs.length;
    // 判断炸弹 u 能否引爆炸弹 v
    const isConnected = (u, v) => {
        const dx = bombs[u][0] - bombs[v][0];
        const dy = bombs[u][1] - bombs[v][1];
        return bombs[u][2] * bombs[u][2] >= dx * dx + dy * dy;
    };
    
    // 维护引爆关系有向图
    const edges = new Map();
    for (let i = 0; i < n; ++i) {
        for (let j = 0; j < n; ++j) {
            if (i !== j && isConnected(i, j)) {
                if (!edges.has(i)) edges.set(i, []);
                edges.get(i).push(j);
            }
        }
    }
    let res = 0; // 最多引爆数量
    for (let i = 0; i < n; ++i) {
        // 遍历每个炸弹，广度优先搜索计算该炸弹可引爆的数量，并维护最大值
        const visited = Array(n).fill(0);
        let cnt = 1;
        const q = [i];
        visited[i] = 1;
        while (q.length > 0) {
            const cidx = q.shift();
            for (const nidx of edges.get(cidx) || []) {
                if (visited[nidx]) continue;
                ++cnt;
                q.push(nidx);
                visited[nidx] = 1;
            }
        }
        res = Math.max(res, cnt);
    }
    return res;
};



const mod = 1e9 + 7;
const inf = 0x3f3f3f3f;

var sumOfPowers = function(nums, k) {
    const n = nums.length;
    let res = 0;
    const d = Array.from({ length: n }, () => Array.from({ length: k + 1 }, () => new Map()));
    nums.sort((a, b) => a - b);
    
    for (let i = 0; i < n; i++) {
        d[i][1].set(inf, 1);
        for (let j = 0; j < i; j++) {
            const diff = Math.abs(nums[i] - nums[j]);
            for (let p = 2; p <= k; p++) {
                for (const [v, cnt] of d[j][p - 1].entries()) {
                    const key = Math.min(diff, v);
                    d[i][p].set(key, (d[i][p].get(key) || 0) + cnt % mod);
                }
            }
        }
        for (const [v, cnt] of d[i][k].entries()) {
            res = (res + v * cnt % mod) % mod;
        }
    }
    return res;
};


/**
 * @param {number[]} nums
 * @param {number[]} moveFrom
 * @param {number[]} moveTo
 * @return {number[]}
 */
var relocateMarbles = function(nums, moveFrom, moveTo) {
    
};

var relocateMarbles = function(nums, moveFrom, moveTo) {
    let mp = new Map();
    let ans = [];

    nums.forEach(num => mp.set(num, true));
    for (let i = 0; i < moveFrom.length; i++) {
        mp.delete(moveFrom[i]);
        mp.set(moveTo[i], true);
    }
    mp.forEach((_, key) => ans.push(key));
    ans.sort((a, b) => a - b);
    return ans;
};



var minimumOperations = function(num) {
    let n = num.length;
    let find0 = false, find5 = false;
    for (let i = n - 1; i >= 0; i--) {
        if (num[i] === '0' || num[i] === '5') {
            if (find0) {
                return n - i - 2;
            }
            if (num[i] === '0') {
                find0 = true;
            } else {
                find5 = true;
            }
        } else if (num[i] === '2' || num[i] === '7') {
            if (find5) {
                return n - i - 2;
            }
        }
    }
    if (find0) {
        return n - 1;
    }
    return n;
};



/**
 * @param {number[]} nums
 * @return {number}
 */
var findValueOfPartition = function(nums) {
    
};


var findValueOfPartition = function(nums) {
    nums.sort((x, y) => x - y);
    let res = Infinity;
    for (let i = 1; i < nums.length; i++) {
        res = Math.min(res, nums[i] - nums[i - 1]);
    }
    return res;
};

var getSmallestString = function(s, k) {
    s = s.split('');
    for (let i = 0; i < s.length; ++i) {
        let dis = Math.min(s[i].charCodeAt(0) - 'a'.charCodeAt(0), 'z'.charCodeAt(0) - s[i].charCodeAt(0) + 1);
        if (dis <= k) {
            s[i] = 'a';
            k -= dis;
        } else {
            s[i] = String.fromCharCode(s[i].charCodeAt(0) - k);
            break;
        }
    }
    return s.join('');
};



var fallingSquares = function(positions) {
    const n = positions.length;
    const heights = [];
    for (let i = 0; i < n; i++) {
        let left1 = positions[i][0], right1 = positions[i][0] + positions[i][1] - 1;
        let height = positions[i][1];
        for (let j = 0; j < i; j++) {
            let left2 = positions[j][0], right2 = positions[j][0] + positions[j][1] - 1;
            if (right1 >= left2 && right2 >= left1) {
                height = Math.max(height, heights[j] + positions[i][1]);
            }
        }
        heights.push(height);
    }
    for (let i = 1; i < n; i++) {
        heights.splice(i, 1, Math.max(heights[i], heights[i - 1]));
    }
    return heights;
};


/**
 * @param {number[][]} variables
 * @param {number} target
 * @return {number[]}
 */
var getGoodIndices = function(variables, target) {
    
};


var getGoodIndices = function(variables, target) {
    const ans = [];
    for (let i = 0; i < variables.length; i++) {
        const v = variables[i];
        if (powMod(powMod(v[0], v[1], 10), v[2], v[3]) === target) {
            ans.push(i);
        }
    }
    return ans;
};

function powMod(x, y, mod) {
    let res = 1;
    while (y > 0) {
        if ((y & 1) === 1) {
            res = (res * x) % mod;
        }
        x = (x * x) % mod;
        y >>= 1;
    }
    return res;
}


/**
 * @param {number[][]} points
 * @param {number} w
 * @return {number}
 */
var minRectanglesToCoverPoints = function(points, w) {
    
};





var minRectanglesToCoverPoints = function(points, w) {
    points.sort((a, b) => a[0] - b[0]);
    let res = 0;
    let bound = -1;
    for (let p of points) {
        if (p[0] > bound) {
            bound = p[0] + w;
            res++;
        }
    }
    return res;
};


var numberOfAlternatingGroups = function(colors, k) {
    const n = colors.length;
    let res = 0, cnt = 1;
    for (let i = -k + 2; i < n; i++) {
        if (colors[(i + n) % n] !== colors[(i - 1 + n) % n]) {
            cnt++;
        } else {
            cnt = 1;
        }
        if (cnt >= k) {
            res++;
        }
    }
    return res;
};




