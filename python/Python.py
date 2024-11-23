# One-dimensional dynamic programming
# Longest Increasing Subsequence

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums:
            return 0
        dp = []
        for i in range(len(nums)):
            dp.append(1)
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)


# Dynamic programming.
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums: return 0
        dp = [1] * len(nums)
        for i in range(len(nums)):
            for j in range(i):
                if nums[j] < nums[i]: # 如果要求非严格递增，将此行 '<' 改为 '<=' 即可。
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)

# Arrays and Strings
# Remove Duplicates from Sorted Array II

class Solution(object):
    def removeDuplicates(self, nums):
        slow = 0
        for fast in range(len(nums)):
            if slow < 2 or nums[fast] != nums[slow - 2]:
                nums[slow] = nums[fast]
                slow += 1
        return slow

#//Rotate Array

# 注：请勿使用切片，会产生额外空间
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        def reverse(i: int, j: int) -> None:
            while i < j:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
                j -= 1

        n = len(nums)
        k %= n  # 轮转 k 次等于轮转 k%n 次
        reverse(0, n - 1)
        reverse(0, k - 1)
        reverse(k, n - 1)


# Best Time to Buy and Sell Stock II

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        for i in range(1, len(prices)):
            tmp = prices[i] - prices[i - 1]
            if tmp > 0: profit += tmp
        return profit

# Two Sum II - Input Array Is Sorted
# Bidirectional double pointer

class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left = 0
        right = len(numbers) - 1
        while True:
            s = numbers[left] + numbers[right]
            if s == target:
                return [left + 1, right + 1]  # 题目要求下标从 1 开始
            if s > target:
                right -= 1
            else:
                left += 1

class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        i, j = 0, len(numbers) - 1
        while i < j:
            s = numbers[i] + numbers[j]
            if s > target: j -= 1
            elif s < target: i += 1
            else: return i + 1, j + 1
        return []


# Substring with Concatenation of All Words
# Sliding Window

class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        word_len = len(words[0])
        word_num = len(words)
        window = word_len * word_num
        ans = []
        # 需要的各单词次数
        cnt = {word:0 for word in words}
        word_cnt = cnt.copy()
        for word in words:
            word_cnt[word] += 1
        
        start = 0
        while start < len(s) - window + 1:
            # 每轮循环实际的各单词次数
            tmp_cnt = cnt.copy()
            for i in range(start, start+window, word_len):
                tmp_word = s[i:i+word_len]
                if tmp_word in tmp_cnt:
                    tmp_cnt[tmp_word] += 1
                else:
                    break
            # 如果单词次数一直即为一个解
            if tmp_cnt == word_cnt:
                ans.append(start)
            start += 1
        return ans

# Minimum Window Substring
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        ans_left, ans_right = -1, len(s)
        cnt = defaultdict(int)  # 比 Counter 更快
        for c in t:
            cnt[c] += 1
        less = len(cnt)  # 有 less 种字母的出现次数 < t 中的字母出现次数

        left = 0
        for right, c in enumerate(s):  # 移动子串右端点
            cnt[c] -= 1  # 右端点字母移入子串
            if cnt[c] == 0:
                # 原来窗口内 c 的出现次数比 t 的少，现在一样多
                less -= 1
            while less == 0:  # 涵盖：所有字母的出现次数都是 >=
                if right - left < ans_right - ans_left:  # 找到更短的子串
                    ans_left, ans_right = left, right  # 记录此时的左右端点
                x = s[left]  # 左端点字母
                if cnt[x] == 0:
                    # x 移出窗口之前，检查出现次数，
                    # 如果窗口内 x 的出现次数和 t 一样，
                    # 那么 x 移出窗口后，窗口内 x 的出现次数比 t 的少
                    less += 1
                cnt[x] += 1  # 左端点字母移出子串
                left += 1
        return "" if ans_left < 0 else s[ans_left: ans_right + 1]




class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeTwoLists(self, list1: ListNode, list2: ListNode) -> ListNode:
        # 创建一个虚拟头节点，方便构造合并后的链表
        dummy = ListNode()
        current = dummy

        # 遍历两个链表，将节点按顺序合并
        while list1 and list2:
            if list1.val < list2.val:
                current.next = list1
                list1 = list1.next
            else:
                current.next = list2
                list2 = list2.next
            current = current.next

        # 如果其中一个链表还有剩余节点，将其连接到新链表的末尾
        if list1:
            current.next = list1
        if list2:
            current.next = list2

        return dummy.next

# 示例驱动代码
if __name__ == "__main__":
    # 创建测试链表
    list1 = ListNode(1, ListNode(2, ListNode(4)))
    list2 = ListNode(1, ListNode(3, ListNode(4)))

    # 实例化 Solution 并调用方法
    solution = Solution()
    merged_head = solution.mergeTwoLists(list1, list2)

    # 打印合并后的链表
    current = merged_head
    while current:
        print(current.val, end=" -> ")
        current = current.next
    print("None")


class Node:
    def __init__(self, val: int, next: 'Node' = None, random: 'Node' = None):
        self.val = val
        self.next = next
        self.random = random

class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head:
            return None

        # Step 1: Create a copy of each node and insert it next to the original node.
        current = head
        while current:
            copy_node = Node(current.val)
            copy_node.next = current.next
            current.next = copy_node
            current = copy_node.next

        # Step 2: Set the random pointers for the copied nodes.
        current = head
        while current:
            if current.random:
                current.next.random = current.random.next
            current = current.next.next

        # Step 3: Separate the original list from the copied list.
        current = head
        copy_head = head.next
        while current:
            copy_node = current.next
            current.next = copy_node.next
            current = current.next
            if copy_node.next:
                copy_node.next = copy_node.next.next

        return copy_head

# 示例驱动代码
if __name__ == "__main__":
    # 创建测试链表
    node1 = Node(7)
    node2 = Node(13)
    node3 = Node(11)
    node4 = Node(10)

    # 设置节点的 next 和 random 指针
    node1.next = node2
    node2.next = node3
    node3.next = node4
    node2.random = node1  # node2 的 random 指向 node1
    node4.random = node3  # node4 的 random 指向 node3

    # 使用 Solution 类的 copyRandomList 方法
    solution = Solution()
    copy_head = solution.copyRandomList(node1)

    # 打印复制后的链表
    current = copy_head
    while current:
        random_val = current.random.val if current.random else "None"
        print(f"Node(val={current.val}, random={random_val})")
        current = current.next


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
        if not head or left == right:
            return head

        # 创建一个虚拟节点，以简化操作，指向头节点
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy

        # Step 1: 找到 left 节点的前一个节点
        for _ in range(left - 1):
            prev = prev.next

        # Step 2: 开始反转 left 到 right 的节点
        current = prev.next
        next_node = None

        for _ in range(right - left):
            next_node = current.next
            current.next = next_node.next
            next_node.next = prev.next
            prev.next = next_node

        return dummy.next

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        if not head or k == 1:
            return head

        # Step 1: Count the total number of nodes in the linked list
        count = 0
        current = head
        while current:
            count += 1
            current = current.next

        # Step 2: Use a dummy node to simplify the reversing process
        dummy = ListNode(0)
        dummy.next = head
        prev_group_end = dummy

        # Step 3: Reverse nodes in k-groups
        while count >= k:
            current = prev_group_end.next
            next_node = current.next

            # Reverse k nodes
            for _ in range(1, k):
                current.next = next_node.next
                next_node.next = prev_group_end.next
                prev_group_end.next = next_node
                next_node = current.next

            # Move the pointer for the end of the previous group
            prev_group_end = current
            count -= k

        return dummy.next



class Solution:
    def sumNumbers(self, root):
        def dfs(node, current_sum):
            if not node:
                return 0
            
            # Update current path sum
            current_sum = current_sum * 10 + node.val
            
            # If leaf node, return the path sum
            if not node.left and not node.right:
                return current_sum
            
            # Recursively sum left and right subtrees
            return dfs(node.left, current_sum) + dfs(node.right, current_sum)
        
        return dfs(root, 0)
    

    class Solution:
    def maxPathSum(self, root):
        self.max_sum = float('-inf')
        
        def dfs(node):
            if not node:
                return 0
            
            # Recursively compute max path sum for left and right subtrees
            left_max = max(dfs(node.left), 0)
            right_max = max(dfs(node.right), 0)
            
            # Compute max path sum through current node
            current_max_path = node.val + left_max + right_max
            
            # Update global max path sum
            self.max_sum = max(self.max_sum, current_max_path)
            
            # Return max path sum that can be extended to parent
            return node.val + max(left_max, right_max)
        
        dfs(root)
        return self.max_sum
    
class BSTIterator:
    def __init__(self, root):
        # Initialize stack to store nodes
        self.stack = []
        
        # Helper function to push all left nodes onto stack
        def pushLeft(node):
            while node:
                self.stack.append(node)
                node = node.left
        
        # Initialize stack with leftmost path
        pushLeft(root)
    
    def hasNext(self):
        # If stack is not empty, we have more nodes to process
        return len(self.stack) > 0
    
    def next(self):
        # Get next node from top of stack
        current = self.stack.pop()
        
        # If current node has right child,
        # push all left nodes of right subtree onto stack
        if current.right:
            temp = current.right
            while temp:
                self.stack.append(temp)
                temp = temp.left
        
        return current.val
    



class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def getMinimumDifference(self, root: TreeNode) -> int:
        # Initialize variables to store the previous value and the minimum difference
        prev = None
        min_diff = float('inf')

        # Helper function for in-order traversal
        def in_order(node):
            nonlocal prev, min_diff
            if node is None:
                return
            # Traverse the left subtree
            in_order(node.left)
            # Process the current node
            if prev is not None:
                min_diff = min(min_diff, node.val - prev)
            prev = node.val
            # Traverse the right subtree
            in_order(node.right)

        # Start in-order traversal from the root
        in_order(root)
        return min_diff

# Example usage:
# root = TreeNode(4)
# root.left = TreeNode(2)
# root.right = TreeNode(6)
# root.left.left = TreeNode(1)
# root.left.right = TreeNode(3)
# solution = Solution()
# print(solution.getMinimumDifference(root))  # Output: 1




from collections import defaultdict, deque

class Solution:
    def findOrder(self, numCourses, prerequisites):
        """
        Returns the ordering of courses you should take to finish all courses.
        
        Args:
        numCourses (int): The total number of courses.
        prerequisites (list[list[int]]): The prerequisites for each course.
        
        Returns:
        list[int]: The ordering of courses. If it's impossible to finish all courses, returns an empty list.
        """
        
        # Create a graph and in-degree map
        graph = defaultdict(list)
        in_degree = {i: 0 for i in range(numCourses)}
        
        # Build the graph and in-degree map
        for course, prerequisite in prerequisites:
            graph[prerequisite].append(course)
            in_degree[course] += 1
        
        # Initialize a queue with courses that have no prerequisites
        queue = deque([course for course in in_degree if in_degree[course] == 0])
        
        # Initialize the result list
        result = []
        
        # Perform DFS
        while queue:
            course = queue.popleft()
            result.append(course)
            
            # Decrease the in-degree of neighboring courses
            for neighbor in graph[course]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # If the result list has the correct length, return it
        if len(result) == numCourses:
            return result
        else:
            return []
        


# 秒出模型： sambanova meta llama 70b instruct
        

from collections import deque

class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: list[str]) -> int:
        """
        Returns the number of words in the shortest transformation sequence from beginWord to endWord.
        
        Args:
        beginWord (str): The starting word.
        endWord (str): The target word.
        wordList (list[str]): A list of words that can be used in the transformation sequence.
        
        Returns:
        int: The number of words in the shortest transformation sequence, or 0 if no such sequence exists.
        """
        
        # Create a set of words for efficient lookups
        word_set = set(wordList)
        
        # If the end word is not in the word list, return 0
        if endWord not in word_set:
            return 0
        
        # Initialize a queue with the starting word and its level
        queue = deque([(beginWord, 1)])
        
        # Perform BFS
        while queue:
            word, level = queue.popleft()
            
            # If the current word is the end word, return its level
            if word == endWord:
                return level
            
            # Generate all possible words by changing one character at a time
            for i in range(len(word)):
                for char in 'abcdefghijklmnopqrstuvwxyz':
                    next_word = word[:i] + char + word[i + 1:]
                    
                    # If the next word is in the word set, add it to the queue and remove it from the word set
                    if next_word in word_set:
                        queue.append((next_word, level + 1))
                        word_set.remove(next_word)
        
        # If no sequence is found, return 0
        return 0

# Example usage:
solution = Solution()
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log","cog"]
print(solution.ladderLength(beginWord, endWord, wordList))  # Output: 5



class TrieNode:
    """A node in the Trie data structure."""
    
    def __init__(self):
        # Initialize a dictionary to store child nodes
        self.children = {}
        
        # Initialize a boolean to mark the end of a word
        self.is_end_of_word = False


class Trie:
    """A Trie data structure."""
    
    def __init__(self):
        # Initialize the root node
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """
        Inserts the string word into the trie.
        
        Args:
        word (str): The word to be inserted.
        """
        
        # Start at the root node
        node = self.root
        
        # Iterate over each character in the word
        for char in word:
            # If the character is not in the node's children, add it
            if char not in node.children:
                node.children[char] = TrieNode()
            
            # Move to the child node
            node = node.children[char]
        
        # Mark the end of the word
        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        """
        Returns true if the string word is in the trie (i.e., was inserted before), and false otherwise.
        
        Args:
        word (str): The word to be searched.
        
        Returns:
        bool: True if the word is in the trie, false otherwise.
        """
        
        # Start at the root node
        node = self.root
        
        # Iterate over each character in the word
        for char in word:
            # If the character is not in the node's children, return False
            if char not in node.children:
                return False
            
            # Move to the child node
            node = node.children[char]
        
        # Return True if the word is marked as the end of a word, False otherwise
        return node.is_end_of_word

    def startsWith(self, prefix: str) -> bool:
        """
        Returns true if there is a previously inserted string word that has the prefix prefix, and false otherwise.
        
        Args:
        prefix (str): The prefix to be searched.
        
        Returns:
        bool: True if a word with the prefix exists, false otherwise.
        """
        
        # Start at the root node
        node = self.root
        
        # Iterate over each character in the prefix
        for char in prefix:
            # If the character is not in the node's children, return False
            if char not in node.children:
                return False
            
            # Move to the child node
            node = node.children[char]
        
        # If we've reached this point, a word with the prefix exists
        return True


# Example usage:
trie = Trie()
trie.insert("apple")
trie.insert("app")
trie.insert("banana")

print(trie.search("apple"))  # Output: True
print(trie.search("app"))    # Output: True
print(trie.search("banana")) # Output: True
print(trie.search("ban"))    # Output: False

print(trie.startsWith("app"))  # Output: True
print(trie.startsWith("ban"))  # Output: True
print(trie.startsWith("ora"))  # Output: False



class TrieNode:
    """A node in the Trie data structure."""
    
    def __init__(self):
        # Initialize the node with an empty dictionary to store children
        self.children = {}
        # Initialize a flag to mark the end of a word
        self.is_end_of_word = False


class WordDictionary:
    """A dictionary of words that supports adding new words and finding if a string matches any previously added string."""
    
    def __init__(self):
        # Initialize the Trie with a root node
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        """Adds a word to the data structure."""
        
        # Start at the root node
        node = self.root
        # Iterate over each character in the word
        for char in word:
            # If the character is not in the node's children, add it
            if char not in node.children:
                node.children[char] = TrieNode()
            # Move to the child node
            node = node.children[char]
        # Mark the end of the word
        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        """Returns true if there is any string in the data structure that matches word or false otherwise."""
        
        # Start at the root node
        return self._search(self.root, word)

    def _search(self, node: TrieNode, word: str) -> bool:
        """Recursively searches for a word in the Trie."""
        
        # Iterate over each character in the word
        for i, char in enumerate(word):
            # If the character is a dot, recursively search all children
            if char == '.':
                return any(self._search(child, word[i+1:]) for child in node.children.values())
            # If the character is not in the node's children, return False
            elif char not in node.children:
                return False
            # Move to the child node
            node = node.children[char]
        # Return True if the node marks the end of a word
        return node.is_end_of_word


# Example usage:
wordDictionary = WordDictionary()
wordDictionary.addWord("bad")
wordDictionary.addWord("dad")
wordDictionary.addWord("mad")
print(wordDictionary.search("pad"))  # Returns False
print(wordDictionary.search("bad"))  # Returns True
print(wordDictionary.search(".ad"))  # Returns True
print(wordDictionary.search("b.."))  # Returns True


class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = None

class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        if not board or not words:
            return []

        m, n = len(board), len(board[0])
        root = TrieNode()
        for word in words:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.word = word

        res = set()
        def dfs(i, j, node):
            if node.word:
                res.add(node.word)
            if i < 0 or i >= m or j < 0 or j >= n or board[i][j] not in node.children:
                return
            temp = board[i][j]
            board[i][j] = '#'
            for x, y in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                dfs(x, y, node.children[temp])
            board[i][j] = temp

        for i in range(m):
            for j in range(n):
                dfs(i, j, root)

        return list(res)


class Solution:
    def letterCombinations(self, digits: str):
        """Returns all possible letter combinations that the number could represent."""
        
        # Create a dictionary to map digits to letters
        phone_mapping = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz"
        }

        def backtrack(combination, next_digits):
            """Recursively generates all possible combinations."""
            
            # If there are no more digits to process, add the combination to the result
            if len(next_digits) == 0:
                result.append(combination)
            # Otherwise, process the next digit
            else:
                # Get the letters corresponding to the next digit
                for letter in phone_mapping[next_digits[0]]:
                    # Recursively generate combinations with the current letter
                    backtrack(combination + letter, next_digits[1:])

        # Initialize the result list
        result = []
        # If the input is not empty, start the backtracking process
        if digits:
            backtrack("", digits)
        # Return the result
        return result


# Example usage:
solution = Solution()
print(solution.letterCombinations("23"))  # Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"]
print(solution.letterCombinations(""))  # Output: []
print(solution.letterCombinations("2"))  # Output: ["a", "b", "c"]


class Solution:
    def permute(self, nums: list[int]):
        """Returns all possible permutations of the input array."""
        
        def backtrack(start, end):
            """Recursively generates all permutations."""
            
            # If we've reached the end of the array, add the current permutation to the result
            if start == end:
                result.append(nums[:])
            # Otherwise, try swapping each element with the current start element
            else:
                for i in range(start, end):
                    # Swap the current start element with the i-th element
                    nums[start], nums[i] = nums[i], nums[start]
                    # Recursively generate permutations with the swapped elements
                    backtrack(start + 1, end)
                    # Swap the elements back to restore the original array
                    nums[start], nums[i] = nums[i], nums[start]

        # Initialize the result list
        result = []
        # Start the backtracking process
        backtrack(0, len(nums))
        # Return the result
        return result



# llama 3.2 3b


# Example usage:
solution = Solution()
print(solution.permute([1, 2, 3]))  # Output: [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
print(solution.permute([0, 1]))  # Output: [[0, 1], [1, 0]]
print(solution.permute([1]))  # Output: [[1]]


class Solution:
    def combinationSum(self, candidates, target):
        """
        Returns a list of all unique combinations of candidates where the chosen numbers sum to target.

        Args:
            candidates (list): A list of distinct integers.
            target (int): The target integer.

        Returns:
            list: A list of lists, where each sublist is a unique combination of candidates that sum to target.
        """
        def backtrack(remain, comb, start):
            # If the remaining sum is zero, it means we have found a valid combination
            if remain == 0:
                result.append(list(comb))
                return
            # If the remaining sum is negative, it means the current combination is not valid
            elif remain < 0:
                return
            # Iterate over the candidates array starting from the start index
            for i in range(start, len(candidates)):
                # Add the current candidate to the current combination
                comb.append(candidates[i])
                # Recursively call the backtrack function with the updated remaining sum and combination
                backtrack(remain - candidates[i], comb, i)
                # Remove the last added candidate from the current combination (backtracking)
                comb.pop()

        # Sort the candidates array in ascending order
        candidates.sort()
        result = []
        backtrack(target, [], 0)
        return result

# Example usage:
solution = Solution()
candidates = [2, 3, 5]
target = 8
print(solution.combinationSum(candidates, target))



# 405
class Solution:
    def totalNQueens(self, n: int) -> int:
        """
        Returns the number of distinct solutions to the n-queens puzzle.

        :param n: The size of the chessboard (n x n).
        :return: The number of distinct solutions.
        """

        def is_safe(board, row, col):
            """
            Checks if it is safe to place a queen at the given position.

            :param board: The current state of the board.
            :param row: The row to check.
            :param col: The column to check.
            :return: True if it is safe, False otherwise.
            """
            for i in range(row):
                if board[i] == col or \
                    board[i] - i == col - row or \
                    board[i] + i == col + row:
                    return False
            return True

        def backtrack(board, row):
            """
            Recursively tries to place queens on the board.

            :param board: The current state of the board.
            :param row: The current row.
            :return: The number of distinct solutions.
            """
            if row == n:
                return 1
            count = 0
            for col in range(n):
                if is_safe(board, row, col):
                    board[row] = col
                    count += backtrack(board, row + 1)
            return count

        board = [-1] * n
        return backtrack(board, 0)
    

    
#1b

class Solution:
    def generateParenthesis(self, n):
        """
        Generates all combinations of well-formed parentheses for a given number of pairs.

        Args:
            n (int): The number of pairs of parentheses.

        Returns:
            list: A list of strings, each representing a combination of well-formed parentheses.
        """
        def backtrack(open_count, close_count, path):
            # If the path is well-formed, add it to the result list
            if len(path) == 2 * n:
                result.append(path)
                return

            # If the number of open parentheses is less than n, add an open parenthesis
            if open_count < n:
                backtrack(open_count + 1, close_count, path + "(")

            # If the number of close parentheses is less than the number of open parentheses, add a close parenthesis
            if close_count < open_count:
                backtrack(open_count, close_count + 1, path + ")")

        result = []
        backtrack(0, 0, "")
        return result

# Example usage:
n = 3
combinations = Solution().generateParenthesis(n)
for combination in combinations:
    print(combination)




class Solution:
    def ladderLength(self, beginWord, endWord, wordList):
        wordList = set(wordList)
        queue = [(beginWord, 1)]
        while queue:
            word, length = queue.pop(0)
            if word == endWord:
                return length
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    next_word = word[:i] + c + word[i+1:]
                    if next_word in wordList:
                        wordList.remove(next_word)
                        queue.append((next_word, length + 1))
        return 0

    def exist(self, board, word):
        if not board:
            return False
        for i in range(len(board)):
            for j in range(len(board[0])):
                if self.dfs(board, i, j, word):
                    return True
        return False

    def dfs(self, board, i, j, word):
        if len(word) == 0: 
            return True
        if i<0 or i>=len(board) or j<0 or j>=len(board[0]) or word[0]!=board[i][j]:
            return False
        tmp = board[i][j]  
        board[i][j] = "#"  
        res = self.dfs(board, i+1, j, word[1:]) or self.dfs(board, i-1, j, word[1:]) \
        or self.dfs(board, i, j+1, word[1:]) or self.dfs(board, i, j-1, word[1:])
        board[i][j] = tmp
        return res


# llama 3.2 90b 

class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        # Base case: if the input array is empty, return None
        if not nums:
            return None

        # Calculate the middle index of the array
        mid = len(nums) // 2

        # Create a new TreeNode with the value of the middle element
        root = TreeNode(nums[mid])

        # Recursively construct the left and right subtrees
        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid+1:])

        # Return the root node of the height-balanced binary search tree
        return root



# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head

        # Split the linked list into two halves
        mid = self.getMiddle(head)
        midNext = mid.next
        mid.next = None

        # Recursively sort the two halves
        left = self.sortList(head)
        right = self.sortList(midNext)

        # Merge the two sorted halves
        return self.merge(left, right)

    def getMiddle(self, head: ListNode) -> ListNode:
        slow = head
        fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        return slow

    def merge(self, left: ListNode, right: ListNode) -> ListNode:
        dummy = ListNode(0)
        current = dummy
        while left and right:
            if left.val < right.val:
                current.next = left
                left = left.next
            else:
                current.next = right
                right = right.next
            current = current.next
        if left:
            current.next = left
        elif right:
            current.next = right
        return dummy.next



class Node:
    def __init__(self, val=False, isLeaf=False, topLeft=None, topRight=None, bottomLeft=None, bottomRight=None):
        self.val = val
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight

class Solution:
    def construct(self, grid: List[List[int]]) -> Node:
        def is_all_same(grid):
            return len(set([grid[i][j] for i in range(len(grid)) for j in range(len(grid[0]))])) == 1

        def construct_node(grid):
            if is_all_same(grid):
                return Node(val=grid[0][0], isLeaf=True)
            else:
                n = len(grid)
                mid = n // 2
                topLeft = [row[:mid] for row in grid[:mid]]
                topRight = [row[mid:] for row in grid[:mid]]
                bottomLeft = [row[:mid] for row in grid[mid:]]
                bottomRight = [row[mid:] for row in grid[mid:]]
                return Node(isLeaf=False,
                             topLeft=construct_node(topLeft),
                             topRight=construct_node(topRight),
                             bottomLeft=construct_node(bottomLeft),
                             bottomRight=construct_node(bottomRight))

        return construct_node(grid)

# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def mergeKLists(self, lists):
        if not lists:
            return None
        if len(lists) == 1:
            return lists[0]
        mid = len(lists) // 2
        left = self.mergeKLists(lists[:mid])
        right = self.mergeKLists(lists[mid:])
        return self.mergeTwoLists(left, right)

    def mergeTwoLists(self, l1, l2):
        dummy = ListNode(0)
        curr = dummy
        while l1 and l2:
            if l1.val < l2.val:
                curr.next = l1
                l1 = l1.next
            else:
                curr.next = l2
                l2 = l2.next
            curr = curr.next
        if l1:
            curr.next = l1
        elif l2:
            curr.next = l2
        return dummy.next
    

class Solution:
    def maxSubArray(self, nums):
        max_sum = float('-inf')
        current_sum = 0

        for num in nums:
            current_sum = max(num, current_sum + num)
            max_sum = max(max_sum, current_sum)

        return max_sum
    

class Solution:
    def maxSubarraySumCircular(self, nums):
        total_sum = sum(nums)
        max_sum = float('-inf')
        current_sum = 0
        min_sum = float('inf')
        current_min_sum = 0

        for num in nums:
            current_sum = max(num, current_sum + num)
            max_sum = max(max_sum, current_sum)

            current_min_sum = min(num, current_min_sum + num)
            min_sum = min(min_sum, current_min_sum)

        if total_sum == min_sum:
            return max_sum
        else:
            return max(max_sum, total_sum - min_sum)
        


class Solution:
    def searchInsert(self, nums, target):
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return left

