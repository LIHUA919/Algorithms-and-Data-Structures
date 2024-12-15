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
    


import heapq

class Solution:
    def findMaximizedCapital(self, k, w, profits, capital):
        """
        Find the maximum capital that can be obtained by selecting at most k distinct projects.

        Args:
        k (int): The maximum number of distinct projects that can be selected.
        w (int): The initial capital.
        profits (list[int]): A list of pure profits for each project.
        capital (list[int]): A list of minimum capital requirements for each project.

        Returns:
        int: The final maximized capital.
        """
        # Combine profits and capital into a list of tuples and sort by capital
        projects = sorted(zip(capital, profits))

        # Initialize a priority queue to store the projects that can be started
        pq = []

        # Initialize the index of the current project
        i = 0

        # Iterate over the projects
        while k > 0:
            # Add projects that can be started to the priority queue
            while i < len(projects) and projects[i][0] <= w:
                # Push the project's profit into the priority queue
                heapq.heappush(pq, -projects[i][1])
                i += 1

            # If no projects can be started, break the loop
            if not pq:
                break

            # Start the project with the highest profit
            w += -heapq.heappop(pq)
            k -= 1

        # Return the final maximized capital
        return w


import heapq

class Solution:
    def kSmallestPairs(self, nums1, nums2, k):
        if not nums1 or not nums2:
            return []

        # Min-heap to store (sum, index in nums1, index in nums2)
        min_heap = []
        for i in range(min(len(nums1), k)):  # Only need the first k pairs
            heapq.heappush(min_heap, (nums1[i] + nums2[0], i, 0))

        result = []
        while k > 0 and min_heap:
            sum, i, j = heapq.heappop(min_heap)
            result.append((nums1[i], nums2[j]))
            k -= 1
            if j + 1 < len(nums2):
                heapq.heappush(min_heap, (nums1[i] + nums2[j + 1], i, j + 1))

        return result



import heapq

class MedianFinder:

    def __init__(self):
        self.max_heap = []
        self.min_heap = []

    def addNum(self, num: int) -> None:
        if not self.max_heap or num <= -self.max_heap[0]:
            heapq.heappush(self.max_heap, -num)
        else:
            heapq.heappush(self.min_heap, num)

        if len(self.max_heap) > len(self.min_heap) + 1:
            heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        elif len(self.min_heap) > len(self.max_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))

    def findMedian(self) -> float:
        if len(self.max_heap) == len(self.min_heap):
            return (-self.max_heap[0] + self.min_heap[0]) / 2
        else:
            return -self.max_heap[0]
        


class Solution:
    def addBinary(self, a: str, b: str) -> str:
        result = ''
        carry = 0
        i, j = len(a) - 1, len(b) - 1

        while i >= 0 or j >= 0 or carry:
            if i >= 0:
                carry += int(a[i])
                i -= 1
            if j >= 0:
                carry += int(b[j])
                j -= 1
            result = str(carry % 2) + result
            carry //= 2

        return result
    

class Solution:
    def reverseBits(self, n: int) -> int:
        result = 0
        for _ in range(32):
            result = (result << 1) | (n & 1)
            n >>= 1
        return result
    

class Solution:
    def hammingWeight(self, n: int) -> int:
        return n.bit_count()
    

class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        result = 0
        for num in nums:
            result ^= num
        return result
    

class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        ones = twos = 0
        for num in nums:
            twos |= ones & num
            ones ^= num
            threes = ones & twos
            ones &= ~threes
            twos &= ~threes
        return ones
    


class Solution:
    def rangeBitwiseAnd(self, left: int, right: int) -> int:
        shift = 0
        while left < right:
            left >>= 1
            right >>= 1
            shift += 1
        return left << shift
    

class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False
        return str(x) == str(x)[::-1]
    




class Solution:
    def plusOne(self, digits):
        n = len(digits)

        # Process digits from the least significant to the most significant
        for i in range(n - 1, -1, -1):
            digits[i] += 1
            if digits[i] < 10:
                return digits
            digits[i] = 0  # Reset to 0 and carry over

        # If all digits were 9, we need to add a leading 1
        return [1] + digits


class Solution:
    def trailingZeroes(self, n):
        count = 0
        while n >= 5:
            n //= 5
            count += n
        return count


class Solution:
    def mySqrt(self, x):
        if x == 0:
            return 0

        left, right = 1, x  # Start the binary search range
        while left <= right:
            mid = left + (right - left) // 2
            if mid * mid == x:  # Perfect square
                return mid
            elif mid * mid < x:  # Search on the right
                left = mid + 1
            else:  # Search on the left
                right = mid - 1
        return right  # 'right' points to the integer square root



class Solution:
    def myPow(self, x, n):
        if n == 0:
            return 1  # Any number to the power 0 is 1
        if n < 0:
            x = 1 / x  # Invert x for negative exponents
            n = -n

        result = 1
        current_product = x

        while n > 0:
            if n % 2 == 1:  # If n is odd, include current_product in the result
                result *= current_product
            current_product *= current_product  # Square the current product
            n //= 2  # Reduce n by half

        return result

from collections import defaultdict
from math import gcd

class Solution:
    def maxPoints(self, points):
        if len(points) <= 2:
            return len(points)

        def get_slope(p1, p2):
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if dx == 0:  # Vertical line
                return ('inf', 0)
            if dy == 0:  # Horizontal line
                return (0, 'inf')
            d = gcd(dx, dy)  # Reduce fraction
            return (dy // d, dx // d)

        max_points = 0
        for i, p1 in enumerate(points):
            slopes = defaultdict(int)
            overlaps = 0
            current_max = 0
            for j, p2 in enumerate(points):
                if i == j:
                    continue
                if p1 == p2:
                    overlaps += 1
                else:
                    slope = get_slope(p1, p2)
                    slopes[slope] += 1
                    current_max = max(current_max, slopes[slope])
            max_points = max(max_points, current_max + overlaps + 1)

        return max_points


class Solution:
    def climbStairs(self, n):
        if n <= 1:
            return 1
        
        # Start with the base cases
        prev2, prev1 = 1, 1
        
        # Iteratively compute ways
        for _ in range(2, n + 1):
            curr = prev1 + prev2
            prev2, prev1 = prev1, curr
        
        return prev1

# Examples to test the solution
solution = Solution()

# Example 1
n = 2
print(f"Number of ways to climb {n} steps: {solution.climbStairs(n)}")
# Output: 2
# Explanation:
# 1. 1 step + 1 step
# 2. 2 steps

# Example 2
n = 3
print(f"Number of ways to climb {n} steps: {solution.climbStairs(n)}")
# Output: 3
# Explanation:
# 1. 1 step + 1 step + 1 step
# 2. 1 step + 2 steps
# 3. 2 steps + 1 step

# Example 3
n = 5
print(f"Number of ways to climb {n} steps: {solution.climbStairs(n)}")
# Output: 8
# Explanation:
# 1. 1 step + 1 step + 1 step + 1 step + 1 step
# 2. 1 step + 1 step + 1 step + 2 steps
# 3. 1 step + 1 step + 2 steps + 1 step
# 4. 1 step + 2 steps + 1 step + 1 step
# 5. 2 steps + 1 step + 1 step + 1 step
# 6. 1 step + 2 steps + 2 steps
# 7. 2 steps + 1 step + 2 steps
# 8. 2 steps + 2 steps + 1 step



class Solution:
    def rob(self, nums):
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]

        prev2, prev1 = 0, nums[0]  # Initialize rob(i-2) and rob(i-1)

        for i in range(1, len(nums)):
            current = max(nums[i] + prev2, prev1)
            prev2, prev1 = prev1, current

        return prev1

# Examples to test the solution
solution = Solution()

# Example 1
nums = [1, 2, 3, 1]
print(f"Max money that can be robbed from {nums}: {solution.rob(nums)}")
# Output: 4
# Explanation:
# Rob house 1 (1) and house 3 (3), total = 1 + 3 = 4.

# Example 2
nums = [2, 7, 9, 3, 1]
print(f"Max money that can be robbed from {nums}: {solution.rob(nums)}")
# Output: 12
# Explanation:
# Rob house 1 (2), house 3 (9), and house 5 (1), total = 2 + 9 + 1 = 12.

# Example 3
nums = [5]
print(f"Max money that can be robbed from {nums}: {solution.rob(nums)}")
# Output: 5
# Explanation:
# Only one house to rob.

# Example 4
nums = [10, 20, 30]
print(f"Max money that can be robbed from {nums}: {solution.rob(nums)}")
# Output: 40
# Explanation:
# Rob house 1 (10) and house 3 (30), total = 10 + 30 = 40.

# Example 5
nums = []
print(f"Max money that can be robbed from {nums}: {solution.rob(nums)}")
# Output: 0
# Explanation:
# No houses to rob.


class Solution:
    def wordBreak(self, s, wordDict):
        word_set = set(wordDict)  # Convert wordDict to a set for fast lookup
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True  # Base case: empty string can be segmented

        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break

        return dp[n]

# Examples
solution = Solution()

# Example 1
s = "leetcode"
wordDict = ["leet", "code"]
print(f"Can '{s}' be segmented? {solution.wordBreak(s, wordDict)}")
# Output: True

# Example 2
s = "applepenapple"
wordDict = ["apple", "pen"]
print(f"Can '{s}' be segmented? {solution.wordBreak(s, wordDict)}")
# Output: True

# Example 3
s = "catsandog"
wordDict = ["cats", "dog", "sand", "and", "cat"]
print(f"Can '{s}' be segmented? {solution.wordBreak(s, wordDict)}")
# Output: False



class Solution:
    def coinChange(self, coins, amount):
        # Initialize the dp array with infinity for all amounts
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0  # Base case: no coins needed to make amount 0

        # Iterate over all amounts
        for i in range(1, amount + 1):
            for coin in coins:
                if i - coin >= 0:  # If the amount is valid after subtracting the coin
                    dp[i] = min(dp[i], dp[i - coin] + 1)

        return dp[amount] if dp[amount] != float('inf') else -1


# Examples
solution = Solution()

# Example 1
coins = [1, 2, 5]
amount = 11
print(f"Fewest coins to make {amount} with {coins}: {solution.coinChange(coins, amount)}")
# Output: 3

# Example 2
coins = [2]
amount = 3
print(f"Fewest coins to make {amount} with {coins}: {solution.coinChange(coins, amount)}")
# Output: -1

# Example 3
coins = [1]
amount = 0
print(f"Fewest coins to make {amount} with {coins}: {solution.coinChange(coins, amount)}")
# Output: 0

# Example 4
coins = [186, 419, 83, 408]
amount = 6249
print(f"Fewest coins to make {amount} with {coins}: {solution.coinChange(coins, amount)}")
# Output: 20


# Dynamic Programming Solution
class Solution:
    def lengthOfLIS(self, nums):
        n = len(nums)
        dp = [1] * n  # Initialize DP array with 1 (each element is an LIS of length 1)

        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:  # Update dp[i] if nums[i] can extend LIS ending at nums[j]
                    dp[i] = max(dp[i], dp[j] + 1)

        return max(dp)  # The LIS is the maximum value in dp

# Examples
solution = Solution()

# Example 1
nums = [10, 9, 2, 5, 3, 7, 101, 18]
print(f"LIS of {nums}: {solution.lengthOfLIS(nums)}")  # Output: 4

# Example 2
nums = [0, 1, 0, 3, 2, 3]
print(f"LIS of {nums}: {solution.lengthOfLIS(nums)}")  # Output: 4

# Example 3
nums = [7, 7, 7, 7, 7, 7, 7]
print(f"LIS of {nums}: {solution.lengthOfLIS(nums)}")  # Output: 1


class Solution:
    def minimumTotal(self, triangle):
        # Start from the second-to-last row and move upwards
        for row in range(len(triangle) - 2, -1, -1):
            for col in range(len(triangle[row])):
                # Update each element with the minimum path sum
                triangle[row][col] += min(triangle[row + 1][col], triangle[row + 1][col + 1])

        # The top element contains the minimum path sum
        return triangle[0][0]

# Examples
solution = Solution()

# Example 1
triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
print(f"Minimum path sum: {solution.minimumTotal(triangle)}")
# Output: 11

# Example 2
triangle = [[-10]]
print(f"Minimum path sum: {solution.minimumTotal(triangle)}")
# Output: -10


class Solution:
    def minPathSum(self, grid):
        m, n = len(grid), len(grid[0])

        # Update the grid in-place to store the minimum path sums
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue  # Top-left corner
                elif i == 0:
                    grid[i][j] += grid[i][j - 1]  # First row
                elif j == 0:
                    grid[i][j] += grid[i - 1][j]  # First column
                else:
                    grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])  # Transition

        return grid[m - 1][n - 1]


# Examples
solution = Solution()

# Example 1
grid = [[1, 3, 1], [1, 5, 1], [4, 2, 1]]
print(f"Minimum path sum: {solution.minPathSum(grid)}")
# Output: 7

# Example 2
grid = [[1, 2, 3], [4, 5, 6]]
print(f"Minimum path sum: {solution.minPathSum(grid)}")
# Output: 12



class Solution:
    def uniquePathsWithObstacles(self, grid):
        m, n = len(grid), len(grid[0])

        # If the starting or ending cell is an obstacle, return 0
        if grid[0][0] == 1 or grid[m-1][n-1] == 1:
            return 0

        # Initialize the DP table
        dp = [[0] * n for _ in range(m)]
        dp[0][0] = 1  # Start point

        # Fill the DP table
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    dp[i][j] = 0  # Obstacle
                else:
                    if i > 0:
                        dp[i][j] += dp[i-1][j]  # From above
                    if j > 0:
                        dp[i][j] += dp[i][j-1]  # From the left

        return dp[m-1][n-1]


# Examples
solution = Solution()

# Example 1
grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
print(f"Unique paths: {solution.uniquePathsWithObstacles(grid)}")  # Output: 2

# Example 2
grid = [[0, 1], [0, 0]]
print(f"Unique paths: {solution.uniquePathsWithObstacles(grid)}")  # Output: 1


# Expand Around Center Solution
class Solution:
    def longestPalindrome(self, s):
        def expandAroundCenter(left, right):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return s[left + 1:right]

        max_palindrome = ""
        for i in range(len(s)):
            palindrome1 = expandAroundCenter(i, i)  # Odd-length palindromes
            palindrome2 = expandAroundCenter(i, i + 1)  # Even-length palindromes

            if len(palindrome1) > len(max_palindrome):
                max_palindrome = palindrome1
            if len(palindrome2) > len(max_palindrome):
                max_palindrome = palindrome2

        return max_palindrome


# Examples
solution = Solution()

# Example 1
s = "babad"
print(f"Longest palindromic substring of '{s}': {solution.longestPalindrome(s)}")
# Output: "bab" or "aba"

# Example 2
s = "cbbd"
print(f"Longest palindromic substring of '{s}': {solution.longestPalindrome(s)}")
# Output: "bb"

# Example 3
s = "a"
print(f"Longest palindromic substring of '{s}': {solution.longestPalindrome(s)}")
# Output: "a"

# Example 4
s = "ac"
print(f"Longest palindromic substring of '{s}': {solution.longestPalindrome(s)}")
# Output: "a" or "c"


class Solution:
    def isInterleave(self, s1, s2, s3):
        m, n = len(s1), len(s2)
        
        # If lengths don't match, s3 cannot be an interleaving
        if m + n != len(s3):
            return False

        # Initialize DP table
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True  # Base case

        # Fill DP table
        for i in range(m + 1):
            for j in range(n + 1):
                if i > 0 and s1[i - 1] == s3[i + j - 1]:
                    dp[i][j] |= dp[i - 1][j]  # Match character from s1
                if j > 0 and s2[j - 1] == s3[i + j - 1]:
                    dp[i][j] |= dp[i][j - 1]  # Match character from s2

        return dp[m][n]


# Examples
solution = Solution()

# Example 1
s1 = "aab"
s2 = "axy"
s3 = "aaxaby"
print(f"Is '{s3}' an interleaving of '{s1}' and '{s2}'? {solution.isInterleave(s1, s2, s3)}")
# Output: True

# Example 2
s1 = "abc"
s2 = "def"
s3 = "abdecf"
print(f"Is '{s3}' an interleaving of '{s1}' and '{s2}'? {solution.isInterleave(s1, s2, s3)}")
# Output: False

# Example 3
s1 = ""
s2 = "abc"
s3 = "abc"
print(f"Is '{s3}' an interleaving of '{s1}' and '{s2}'? {solution.isInterleave(s1, s2, s3)}")
# Output: True


class Solution:
    def minDistance(self, word1, word2):
        m, n = len(word1), len(word2)

        # Initialize DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Base cases
        for i in range(m + 1):
            dp[i][0] = i  # Deleting all characters from word1
        for j in range(n + 1):
            dp[0][j] = j  # Inserting all characters into word1

        # Fill the DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:  # Characters match
                    dp[i][j] = dp[i - 1][j - 1]
                else:  # Characters don't match
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        return dp[m][n]


# Examples
solution = Solution()

# Example 1
word1 = "horse"
word2 = "ros"
print(f"Edit distance between '{word1}' and '{word2}': {solution.minDistance(word1, word2)}")
# Output: 3

# Example 2
word1 = "intention"
word2 = "execution"
print(f"Edit distance between '{word1}' and '{word2}': {solution.minDistance(word1, word2)}")
# Output: 5



class Solution:
    def maxProfit(self, prices):
        if not prices or len(prices) < 2:
            return 0

        n = len(prices)

        # Calculate the max profit for the first transaction from left to right
        left = [0] * n
        min_price = prices[0]
        for i in range(1, n):
            min_price = min(min_price, prices[i])
            left[i] = max(left[i - 1], prices[i] - min_price)

        # Calculate the max profit for the second transaction from right to left
        right = [0] * n
        max_price = prices[-1]
        for i in range(n - 2, -1, -1):
            max_price = max(max_price, prices[i])
            right[i] = max(right[i + 1], max_price - prices[i])

        # Combine the results from left and right
        max_profit = 0
        for i in range(n):
            max_profit = max(max_profit, left[i] + right[i])

        return max_profit

# Examples
solution = Solution()

# Example 1
prices = [3, 3, 5, 0, 0, 3, 1, 4]
print(f"Maximum profit: {solution.maxProfit(prices)}")  # Output: 6

# Example 2
prices = [1, 2, 3, 4, 5]
print(f"Maximum profit: {solution.maxProfit(prices)}")  # Output: 4

# Example 3
prices = [7, 6, 4, 3, 1]
print(f"Maximum profit: {solution.maxProfit(prices)}")  # Output: 0



class Solution:
    def maxProfit(self, k, prices):
        n = len(prices)
        if not prices or k == 0:
            return 0

        # If k >= n // 2, it's equivalent to unlimited transactions
        if k >= n // 2:
            return sum(max(prices[i + 1] - prices[i], 0) for i in range(n - 1))

        # Initialize DP table
        dp = [[0] * n for _ in range(k + 1)]

        for i in range(1, k + 1):
            maxDiff = -prices[0]  # Maximum difference for current transaction
            for j in range(1, n):
                dp[i][j] = max(dp[i][j - 1], prices[j] + maxDiff)
                maxDiff = max(maxDiff, dp[i - 1][j] - prices[j])

        return dp[k][n - 1]


# Examples
solution = Solution()

# Example 1
k = 2
prices = [2, 4, 1]
print(f"Maximum profit with {k} transactions: {solution.maxProfit(k, prices)}")  # Output: 2

# Example 2
k = 2
prices = [3, 2, 6, 5, 0, 3]
print(f"Maximum profit with {k} transactions: {solution.maxProfit(k, prices)}")  # Output: 7



class Solution:
    def maximalSquare(self, matrix):
        if not matrix or not matrix[0]:
            return 0

        m, n = len(matrix), len(matrix[0])
        dp = [[0] * n for _ in range(m)]
        max_side = 0

        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '1':
                    if i == 0 or j == 0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                    max_side = max(max_side, dp[i][j])

        return max_side * max_side


# Examples
solution = Solution()

# Example 1
matrix = [
    ["1", "0", "1", "0", "0"],
    ["1", "0", "1", "1", "1"],
    ["1", "1", "1", "1", "1"],
    ["1", "0", "0", "1", "0"]
]
print(f"Largest square area: {solution.maximalSquare(matrix)}")  # Output: 4

# Example 2
matrix = [
    ["0", "1"],
    ["1", "0"]
]
print(f"Largest square area: {solution.maximalSquare(matrix)}")  # Output: 1

# Example 3
matrix = [
    ["0"]
]
print(f"Largest square area: {solution.maximalSquare(matrix)}")  # Output: 0


class Solution:
    def wordPattern(self, pattern, s):
        words = s.split()  # Split the string into words

        # If lengths don't match, return False
        if len(pattern) != len(words):
            return False

        char_to_word = {}
        word_to_char = {}

        for char, word in zip(pattern, words):
            if char in char_to_word:
                if char_to_word[char] != word:
                    return False
            else:
                char_to_word[char] = word

            if word in word_to_char:
                if word_to_char[word] != char:
                    return False
            else:
                word_to_char[word] = char

        return True


# Examples
solution = Solution()

# Example 1
pattern = "abba"
s = "dog cat cat dog"
print(f"Does '{s}' follow pattern '{pattern}'? {solution.wordPattern(pattern, s)}")  # Output: True

# Example 2
pattern = "abba"
s = "dog cat cat fish"
print(f"Does '{s}' follow pattern '{pattern}'? {solution.wordPattern(pattern, s)}")  # Output: False

# Example 3
pattern = "aaaa"
s = "dog cat cat dog"
print(f"Does '{s}' follow pattern '{pattern}'? {solution.wordPattern(pattern, s)}")  # Output: False

# Example 4
pattern = "abba"
s = "dog dog dog dog"
print(f"Does '{s}' follow pattern '{pattern}'? {solution.wordPattern(pattern, s)}")  # Output: False



class Solution:
    def intToRoman(self, num):
        # Define the Roman numeral mappings
        values = [
            (1000, 'M'),
            (900, 'CM'),
            (500, 'D'),
            (400, 'CD'),
            (100, 'C'),
            (90, 'XC'),
            (50, 'L'),
            (40, 'XL'),
            (10, 'X'),
            (9, 'IX'),
            (5, 'V'),
            (4, 'IV'),
            (1, 'I')
        ]

        result = []
        for value, symbol in values:
            # Append the symbol for as many times as value fits into num
            while num >= value:
                result.append(symbol)
                num -= value

        return ''.join(result)


# Examples
solution = Solution()

# Example 1
num = 58
print(f"Roman numeral for {num}: {solution.intToRoman(num)}")  # Output: "LVIII"

# Example 2
num = 1994
print(f"Roman numeral for {num}: {solution.intToRoman(num)}")  # Output: "MCMXCIV"

# Example 3
num = 9
print(f"Roman numeral for {num}: {solution.intToRoman(num)}")  # Output: "IX"

class Solution:
    def lengthOfLastWord(self, s):
        length = 0
        i = len(s) - 1

        # Skip trailing spaces
        while i >= 0 and s[i] == ' ':
            i -= 1

        # Count the length of the last word
        while i >= 0 and s[i] != ' ':
            length += 1
            i -= 1

        return length


# Examples
solution = Solution()

# Example 1
s = "Hello World"
print(f"Length of last word in '{s}': {solution.lengthOfLastWord(s)}")  # Output: 5

# Example 2
s = "   fly me   to   the moon  "
print(f"Length of last word in '{s}': {solution.lengthOfLastWord(s)}")  # Output: 4

# Example 3
s = "luffy is still joyboy"
print(f"Length of last word in '{s}': {solution.lengthOfLastWord(s)}")  # Output: 6





class Solution:
    def longestCommonPrefix(self, strs):
        if not strs:
            return ""

        # Start with the first string as the prefix
        prefix = strs[0]

        for string in strs[1:]:
            # Reduce the prefix until it matches the current string
            while string[:len(prefix)] != prefix:
                prefix = prefix[:-1]
                if not prefix:
                    return ""

        return prefix


# Examples
solution = Solution()

# Example 1
strs = ["flower", "flow", "flight"]
print(f"Longest common prefix: '{solution.longestCommonPrefix(strs)}'")  # Output: "fl"

# Example 2
strs = ["dog", "racecar", "car"]
print(f"Longest common prefix: '{solution.longestCommonPrefix(strs)}'")  # Output: ""

# Example 3
strs = ["interview", "interval", "internet"]
print(f"Longest common prefix: '{solution.longestCommonPrefix(strs)}'")  # Output: "inte"


class Solution:
    def reverseWords(self, s):
        # Split the string into words, reverse them, and join with a single space
        return ' '.join(s.split()[::-1])


# Examples
solution = Solution()

# Example 1
s = "the sky is blue"
print(f"Reversed words: '{solution.reverseWords(s)}'")  # Output: "blue is sky the"

# Example 2
s = "  hello world  "
print(f"Reversed words: '{solution.reverseWords(s)}'")  # Output: "world hello"

# Example 3
s = "a good   example"
print(f"Reversed words: '{solution.reverseWords(s)}'")  # Output: "example good a"



class Solution:
    def convert(self, s, numRows):
        if numRows == 1 or numRows >= len(s):
            return s

        # Initialize a list for each row
        rows = [''] * numRows
        current_row = 0
        going_down = False

        for char in s:
            rows[current_row] += char
            # Change direction at the first or last row
            if current_row == 0 or current_row == numRows - 1:
                going_down = not going_down
            current_row += 1 if going_down else -1

        # Concatenate all rows
        return ''.join(rows)


# Examples
solution = Solution()

# Example 1
s = "PAYPALISHIRING"
numRows = 3
print(f"Converted string: {solution.convert(s, numRows)}")  # Output: "PAHNAPLSIIGYIR"

# Example 2
s = "PAYPALISHIRING"
numRows = 4
print(f"Converted string: {solution.convert(s, numRows)}")  # Output: "PINALSIGYAHRPI"

# Example 3
s = "A"
numRows = 1
print(f"Converted string: {solution.convert(s, numRows)}")  # Output: "A"



class Solution:
    def strStr(self, haystack, needle):
        # Sliding Window Implementation
        if not needle:
            return 0

        needle_len = len(needle)
        haystack_len = len(haystack)

        for i in range(haystack_len - needle_len + 1):
            if haystack[i:i + needle_len] == needle:
                return i

        return -1

# Examples
solution = Solution()

# Example 1
haystack = "sadbutsad"
needle = "sad"
print(f"First occurrence of '{needle}' in '{haystack}': {solution.strStr(haystack, needle)}")  # Output: 0

# Example 2
haystack = "leetcode"
needle = "leeto"
print(f"First occurrence of '{needle}' in '{haystack}': {solution.strStr(haystack, needle)}")  # Output: -1

# Example 3
haystack = "hello"
needle = ""
print(f"First occurrence of '{needle}' in '{haystack}': {solution.strStr(haystack, needle)}")  # Output: 0




class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        # Basic length check
        if len(s1) + len(s2) != len(s3):
            return False
        
        # Create DP table
        # dp[i][j] represents if first i chars of s1 and first j chars of s2 
        # can form first (i+j) chars of s3
        dp = [[False] * (len(s2) + 1) for _ in range(len(s1) + 1)]
        
        # Initialize base case
        dp[0][0] = True
        
        # Fill first row (only s2 is used)
        for j in range(1, len(s2) + 1):
            dp[0][j] = dp[0][j-1] and s2[j-1] == s3[j-1]
        
        # Fill first column (only s1 is used)
        for i in range(1, len(s1) + 1):
            dp[i][0] = dp[i-1][0] and s1[i-1] == s3[i-1]
        
        # Fill the rest of the DP table
        for i in range(1, len(s1) + 1):
            for j in range(1, len(s2) + 1):
                # Check if current character from s1 matches
                if dp[i-1][j] and s1[i-1] == s3[i+j-1]:
                    dp[i][j] = True
                # Check if current character from s2 matches
                elif dp[i][j-1] and s2[j-1] == s3[i+j-1]:
                    dp[i][j] = True
        
        # Return final state
        return dp[len(s1)][len(s2)]
    




class Solution:
    def largestComponentSize(self, nums: List[int]) -> int:
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                
        # Find maximum number to determine the size of parent array
        max_num = max(nums)
        parent = list(range(max_num + 1))
        
        # Union numbers with common factors
        for num in nums:
            # Find factors from 2 to sqrt(num)
            for factor in range(2, int(num**0.5) + 1):
                if num % factor == 0:
                    # Union the number with its factors
                    union(num, factor)
                    union(num, num // factor)
        
        # Count the size of each component
        count = {}
        max_component = 0
        
        for num in nums:
            root = find(num)
            count[root] = count.get(root, 0) + 1
            max_component = max(max_component, count[root])
        
        return max_component
    


    class Solution:
    def waysToFillArray(self, queries: List[List[int]]) -> List[int]:
        def factorize(n):
            # Prime factorization of a number
            factors = {}
            d = 2
            while d * d <= n:
                while n % d == 0:
                    factors[d] = factors.get(d, 0) + 1
                    n //= d
                d += 1
            if n > 1:
                factors[n] = factors.get(n, 0) + 1
            return factors

        def count_ways(n, k):
            # If k is 1, only one way to fill the array
            if k == 1:
                return 1
            
            # Prime factorize k
            prime_factors = factorize(k)
            
            # Dynamic programming to compute combinations
            MOD = 10**9 + 7
            
            # Total ways is product of ways for each prime factor
            total_ways = 1
            for prime, count in prime_factors.items():
                # Ways to distribute 'count' prime factors in 'n' slots
                # This is equivalent to stars and bars problem
                total_ways *= comb(n + count - 1, count)
                total_ways %= MOD
            
            return total_ways

        # Combination with memoization
        @lru_cache(None)
        def comb(n, k):
            MOD = 10**9 + 7
            # Base cases
            if k > n:
                return 0
            if k == 0 or k == n:
                return 1
            
            return (comb(n-1, k-1) + comb(n-1, k)) % MOD

        # Process each query
        return [count_ways(n, k) for n, k in queries]
    

class Solution:
    def kthFactor(self, n: int, k: int) -> int:
        # List to store the factors of n
        factors = []

        # Find all factors of n
        for i in range(1, n + 1):
            if n % i == 0:
                factors.append(i)

        # Return the kth factor if it exists, otherwise return -1
        return factors[k - 1] if k <= len(factors) else -1
    
class Solution:
    def maxCoins(self, nums: list[int]) -> int:
        # Add 1 to both ends of nums to handle boundary cases
        nums = [1] + nums + [1]
        n = len(nums)
        
        # DP table
        dp = [[0] * n for _ in range(n)]
        
        # Fill the DP table
        for length in range(2, n):  # Length of the range (start to end)
            for i in range(n - length):  # Start index of the range
                j = i + length  # End index of the range
                for k in range(i + 1, j):  # k is the last balloon burst in range (i, j)
                    dp[i][j] = max(dp[i][j], dp[i][k] + dp[k][j] + nums[i] * nums[k] * nums[j])
        
        # The result is stored in dp[0][n - 1]
        return dp[0][n - 1]


class Solution:
    def maximalRectangle(self, matrix: list[list[str]]) -> int:
        if not matrix or not matrix[0]:
            return 0
        
        rows, cols = len(matrix), len(matrix[0])
        heights = [0] * cols  # Histogram heights
        max_area = 0
        
        def largestRectangleArea(heights):
            stack = []  # Store indices of heights
            max_area = 0
            heights.append(0)  # Add a sentinel value to pop remaining heights
            for i in range(len(heights)):
                while stack and heights[i] < heights[stack[-1]]:
                    h = heights[stack.pop()]
                    width = i if not stack else i - stack[-1] - 1
                    max_area = max(max_area, h * width)
                stack.append(i)
            heights.pop()  # Remove the sentinel value
            return max_area
        







        for row in matrix:
            for col in range(cols):
                # Update heights
                heights[col] = heights[col] + 1 if row[col] == "1" else 0
            
            # Compute max area for the current histogram
            max_area = max(max_area, largestRectangleArea(heights))
        
        return max_area









class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        n = len(nums)
        val = [1] + nums + [1]
        
        @lru_cache(None)
        def solve(left: int, right: int) -> int:
            if left >= right - 1:
                return 0
            
            best = 0
            for i in range(left + 1, right):
                total = val[left] * val[i] * val[right]
                total += solve(left, i) + solve(i, right)
                best = max(best, total)
            
            return best

        return solve(0, n + 1)

class Solution:
    def kthFactor(self, n: int, k: int) -> int:
        count = 0
        for factor in range(1, n+1):
            if n % factor == 0:
                count += 1
                if count == k:
                    return factor
        return -1








import math

class Solution:
    def findGCD(self, nums: List[int]) -> int:
        mx, mn = max(nums), min(nums)
        return math.gcd(mx, mn)


class Solution:
    def waysToFillArray(self, queries: List[List[int]]) -> List[int]:
        mod, max_n, max_m = 10**9 + 7, 10**4 + 14, 14
        comb = [[0] * max_m for _ in range(max_n)]
        comb[0][0] = 1

        for i in range(1, max_n):
            comb[i][0] = 1
            for j in range(1, min(i + 1, max_m)):
                comb[i][j] = (comb[i - 1][j - 1] + comb[i - 1][j]) % mod

        ans = []
        for n, k in queries:
            i, tot = 2, 1
            while i * i <= k:
                if k % i == 0:
                    cnt = 0
                    while k % i == 0:
                        k /= i
                        cnt += 1
                    tot = (tot * comb[n + cnt - 1][cnt]) % mod
                i += 1
            # k 自身为质数
            if k > 1:
                tot = tot * n % mod
            ans.append(tot)

        return ans



from collections import deque

class Solution:
    def maxNumber(self, nums1, nums2, k):
        def maxSubarray(nums, l):
            """Get the maximum subarray of length l."""
            stack = []
            drop = len(nums) - l
            for num in nums:
                while drop > 0 and stack and stack[-1] < num:
                    stack.pop()
                    drop -= 1
                stack.append(num)
            return stack[:l]
        
        def mergeArrays(arr1, arr2):
            """Merge two arrays into the largest possible number."""
            arr1, arr2 = deque(arr1), deque(arr2)
            result = []
            while arr1 or arr2:
                if list(arr1) > list(arr2):
                    result.append(arr1.popleft())
                else:
                    result.append(arr2.popleft())
            return result
        
        max_result = []
        m, n = len(nums1), len(nums2)
        
        # Iterate over all valid splits of k
        for i in range(max(0, k - n), min(k, m) + 1):
            sub1 = maxSubarray(nums1, i)
            sub2 = maxSubarray(nums2, k - i)
            merged = mergeArrays(sub1, sub2)
            max_result = max(max_result, merged)
        
        return max_result





class Solution:
    def maxTaskAssign(self, tasks, workers, pills, strength):
        def canComplete(mid):
            """Check if `mid` tasks can be completed."""
            tasks_needed = tasks[:mid]  # Consider only the first `mid` tasks
            workers_available = workers[:]  # Copy workers array
            pill_count = pills
            
            # Assign tasks from the hardest to the easiest
            for task in reversed(tasks_needed):
                # If the strongest available worker can do the task
                if workers_available and workers_available[-1] >= task:
                    workers_available.pop()  # Use the worker without a pill
                elif pill_count > 0:
                    # Use a pill on the weakest possible worker to meet the task
                    idx = findWorker(workers_available, task - strength)
                    if idx == -1:
                        return False  # No worker can complete this task even with a pill
                    workers_available.pop(idx)  # Use the selected worker
                    pill_count -= 1
                else:
                    return False  # Task cannot be completed
            return True
        
        def findWorker(workers, required_strength):
            """Find the weakest worker who can do the task with a pill."""
            # Binary search for the first worker >= required_strength
            low, high = 0, len(workers) - 1
            while low <= high:
                mid = (low + high) // 2
                if workers[mid] >= required_strength:
                    high = mid - 1
                else:
                    low = mid + 1
            return low if low < len(workers) else -1

        # Sort tasks and workers for efficient matching
        tasks.sort()
        workers.sort()
        
        # Binary search on the maximum number of tasks that can be completed
        left, right = 0, min(len(tasks), len(workers))
        result = 0
        
        while left <= right:
            mid = (left + right) // 2
            if canComplete(mid):
                result = mid  # Update result as we can complete `mid` tasks
                left = mid + 1  # Try for more tasks
            else:
                right = mid - 1  # Try for fewer tasks
        
        return result


# failed again 
class Solution:
    def movesToStamp(self, stamp: str, target: str):
        m, n = len(stamp), len(target)
        target = list(target)  # Convert target to a mutable list
        result = []
        stamped = [False] * (n - m + 1)  # Keep track of stamped positions
        
        def canStamp(start):
            """Check if we can stamp `stamp` at `start` in `target`."""
            for i in range(m):
                if target[start + i] != '?' and target[start + i] != stamp[i]:
                    return False
            return True

        def doStamp(start):
            """Apply the stamp at `start` and mark characters as '?'."""
            for i in range(m):
                if target[start + i] != '?':
                    target[start + i] = '?'
            result.append(start)

        totalStamped = 0
        while totalStamped < n:
            stampedInThisRound = False
            for i in range(n - m + 1):
                if not stamped[i] and canStamp(i):
                    doStamp(i)
                    stamped[i] = True
                    stampedInThisRound = True
                    totalStamped += sum(1 for j in range(m) if target[i + j] == '?')
                    break  # Process one stamp per round
            if not stampedInThisRound:
                return []  # If no progress, it's impossible to stamp

        return result[::-1]  # Reverse the result for correct order





import heapq

class Solution:
    def kSmallestPairs(self, nums1, nums2, k):
        # Initialize the min-heap
        min_heap = []
        result = []
        
        # If either nums1 or nums2 is empty, return an empty list
        if not nums1 or not nums2:
            return result
        
        # Start by adding the smallest pairs to the heap
        for i in range(min(k, len(nums1))):  # Limit to k elements in nums1
            heapq.heappush(min_heap, (nums1[i] + nums2[0], i, 0))  # (sum, index_in_nums1, index_in_nums2)
        
        # Extract k smallest pairs
        while min_heap and len(result) < k:
            # Pop the smallest sum pair from the heap
            sum_val, i, j = heapq.heappop(min_heap)
            result.append([nums1[i], nums2[j]])
            
            # If possible, add the next pair from the same row in nums1 but next column in nums2
            if j + 1 < len(nums2):
                heapq.heappush(min_heap, (nums1[i] + nums2[j + 1], i, j + 1))
        
        return result





class Solution:
    def nthUglyNumber(self, n: int) -> int:
        # Initialize the first ugly number
        ugly_numbers = [1]
        
        # Initialize the indices for multiples of 2, 3, and 5
        i2 = i3 = i5 = 0
        
        # Initialize the next multiples of 2, 3, and 5
        next2, next3, next5 = 2, 3, 5
        
        for _ in range(1, n):
            # Get the next ugly number by choosing the smallest from next2, next3, and next5
            next_ugly = min(next2, next3, next5)
            ugly_numbers.append(next_ugly)
            
            # Move the pointer(s) for the respective factors
            if next_ugly == next2:
                i2 += 1
                next2 = ugly_numbers[i2] * 2
            if next_ugly == next3:
                i3 += 1
                next3 = ugly_numbers[i3] * 3
            if next_ugly == next5:
                i5 += 1
                next5 = ugly_numbers[i5] * 5
        
        return ugly_numbers[-1]  # Return the nth ugly number


#falied 

import heapq

class Solution:
    def getSkyline(self, buildings):
        # Step 1: Create events for each building's start and end.
        events = []
        for left, right, height in buildings:
            events.append((left, height, 'start'))  # Start event
            events.append((right, height, 'end'))  # End event
        
        # Step 2: Sort events
        events.sort(key=lambda x: (x[0], -x[1] if x[2] == 'start' else x[1]))
        
        # Step 3: Initialize the result list and the max-heap (priority queue).
        result = []
        max_heap = [(0, float('inf'))]  # (height, right_edge) with ground level as a starting point.
        
        # Step 4: Sweep through the events and process each event.
        for x, height, event_type in events:
            if event_type == 'start':
                # Add building height to the heap
                heapq.heappush(max_heap, (-height, x))
            else:
                # Lazy deletion: just mark the building as ended
                max_heap = [(h, r) for h, r in max_heap if r != x]
                heapq.heapify(max_heap)
            
            # Get the current maximum height
            current_height = -max_heap[0][0]  # Max heap stores negative heights
            
            # If the height has changed, add the point to the result
            if not result or result[-1][1] != current_height:
                result.append([x, current_height])
        
        return result



class Solution:
    def nextPermutation(self, nums):
        # Step 1: Find the pivot (first number that is smaller than its next number from the end)
        n = len(nums)
        i = n - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        
        if i >= 0:
            # Step 2: Find the number just larger than nums[i] in the suffix
            j = n - 1
            while nums[j] <= nums[i]:
                j -= 1
            
            # Step 3: Swap nums[i] and nums[j]
            nums[i], nums[j] = nums[j], nums[i]
        
        # Step 4: Reverse the suffix starting at i + 1
        nums[i + 1:] = reversed(nums[i + 1:])


class Solution:
    def merge(self, nums1, m, nums2, n):
        # Start from the end of nums1 and nums2
        i = m - 1  # Last element in the original nums1
        j = n - 1  # Last element in nums2
        k = m + n - 1  # Last position in nums1

        # Merge nums2 into nums1 starting from the end
        while i >= 0 and j >= 0:
            if nums1[i] > nums2[j]:
                nums1[k] = nums1[i]
                i -= 1
            else:
                nums1[k] = nums2[j]
                j -= 1
            k -= 1

        # If there are remaining elements in nums2, copy them to nums1
        while j >= 0:
            nums1[k] = nums2[j]
            j -= 1
            k -= 1

# Example usage
nums1 = [1, 2, 3, 0, 0, 0]  # Length = m + n
m = 3  # Number of elements in nums1
nums2 = [2, 5, 6]
n = 3  # Number of elements in nums2

# Create an instance of the Solution class and call the merge method
solution = Solution()
solution.merge(nums1, m, nums2, n)

# Output the result
print(nums1)  # Output: [1, 2, 2, 3, 5, 6]


class Solution:
    def rotate(self, nums, k):
        # Step 1: Normalize k to prevent unnecessary rotations
        n = len(nums)
        k = k % n  # Handle cases where k >= n
        
        # Step 2: Reverse the entire array
        self.reverse(nums, 0, n - 1)
        
        # Step 3: Reverse the first k elements
        self.reverse(nums, 0, k - 1)
        
        # Step 4: Reverse the remaining n - k elements
        self.reverse(nums, k, n - 1)
    
    # Helper function to reverse a part of the array
    def reverse(self, nums, start, end):
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1

# Example usage
nums = [1, 2, 3, 4, 5, 6, 7]
k = 3
solution = Solution()
solution.rotate(nums, k)
print(nums)  # Output: [5, 6, 7, 1, 2, 3, 4]





class Solution:
    def compareVersion(self, version1: str, version2: str) -> int:
        # Split both version strings into lists of integers
        v1_parts = list(map(int, version1.split('.')))
        v2_parts = list(map(int, version2.split('.')))
        
        # Compare each corresponding revision
        # We'll pad the shorter version with zeros for comparison
        while len(v1_parts) < len(v2_parts):
            v1_parts.append(0)
        while len(v2_parts) < len(v1_parts):
            v2_parts.append(0)
        
        # Now, both lists are of equal length, compare them element by element
        for i in range(len(v1_parts)):
            if v1_parts[i] < v2_parts[i]:
                return -1
            elif v1_parts[i] > v2_parts[i]:
                return 1
        
        return 0  # If all corresponding parts are equal



class Solution:
    def isPalindrome(self, s: str) -> bool:
        # Step 1: Normalize the string
        filtered_s = ''.join(c.lower() for c in s if c.isalnum())
        
        # Step 2: Check if the string is equal to its reverse
        return filtered_s == filtered_s[::-1]

# Example usage:
solution = Solution()

# Example 1
s1 = "A man, a plan, a canal: Panama"
print(solution.isPalindrome(s1))  # Output: True

# Example 2
s2 = "race a car"
print(solution.isPalindrome(s2))  # Output: False





import heapq

class MedianFinder:
    def __init__(self):
        # Two heaps:
        # max-heap (low) to store the smaller half of the numbers (we store negative to simulate max-heap)
        self.low = []  # Max-heap (simulated using negative numbers)
        # min-heap (high) to store the larger half of the numbers
        self.high = []  # Min-heap
    
    def addNum(self, num: int) -> None:
        # Step 1: Add num to one of the heaps
        if not self.low or num <= -self.low[0]:
            # If num is smaller than or equal to the top of low (max-heap), push it to low
            heapq.heappush(self.low, -num)
        else:
            # Otherwise, push it to high (min-heap)
            heapq.heappush(self.high, num)
        
        # Step 2: Rebalance the heaps
        if len(self.low) > len(self.high) + 1:
            # If low has more than one extra element, move the top of low to high
            heapq.heappush(self.high, -heapq.heappop(self.low))
        elif len(self.high) > len(self.low):
            # If high has more elements, move the top of high to low
            heapq.heappush(self.low, -heapq.heappop(self.high))
    
    def findMedian(self) -> float:
        # Step 3: Find the median
        if len(self.low) > len(self.high):
            # If low has more elements, the median is the top of low
            return -self.low[0]
        else:
            # If both heaps are equal in size, the median is the average of the tops of both heaps
            return (-self.low[0] + self.high[0]) / 2.0

# Example usage:
medianFinder = MedianFinder()

# Add numbers
medianFinder.addNum(1)
print(medianFinder.findMedian())  # Output: 1.0

medianFinder.addNum(2)
print(medianFinder.findMedian())  # Output: 1.5

medianFinder.addNum(3)
print(medianFinder.findMedian())  # Output: 2.0



class Solution:
    def findRepeatedDnaSequences(self, s: str):
        seen = set()  # To track seen substrings
        repeated = set()  # To track substrings that are repeated
        n = len(s)
        
        # If the string length is less than 10, return an empty list
        if n < 10:
            return []
        
        # Iterate through the string and extract 10-character substrings
        for i in range(n - 9):
            substring = s[i:i+10]
            
            if substring in seen:
                # If we've seen this substring before, add it to repeated
                repeated.add(substring)
            else:
                # Otherwise, add it to the seen set
                seen.add(substring)
        
        # Convert the repeated set to a list and return
        return list(repeated)

# Example usage:
solution = Solution()

# Test case 1:
s1 = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"
print(solution.findRepeatedDnaSequences(s1))  # Output: ["AAAAACCCCC", "CCCCCAAAAA"]

# Test case 2:
s2 = "AAAAAAAAAAAAA"
print(solution.findRepeatedDnaSequences(s2))  # Output: ["AAAAAAAAAA"]




# time out

import heapq

class Solution:
    def medianSlidingWindow(self, nums, k):
        def rebalance():
            # Ensure the heaps are balanced in size
            if len(low) > len(high) + 1:
                heapq.heappush(high, -heapq.heappop(low))
            elif len(high) > len(low):
                heapq.heappush(low, -heapq.heappop(high))

        def get_median():
            # If the window size is odd, the median is the top of the max-heap
            if k % 2 == 1:
                return -low[0]
            # If even, the median is the average of the tops of the two heaps
            return (-low[0] + high[0]) / 2.0

        low, high = [], []  # max-heap (low) and min-heap (high)
        result = []

        for i, num in enumerate(nums):
            # Step 1: Add the new number
            if len(low) == 0 or num <= -low[0]:
                heapq.heappush(low, -num)  # Push to max-heap
            else:
                heapq.heappush(high, num)  # Push to min-heap
            rebalance()

            # Step 2: Once we have k elements, calculate the median
            if i >= k - 1:
                result.append(get_median())

                # Step 3: Remove the element sliding out of the window
                out_num = nums[i - k + 1]
                if out_num <= -low[0]:
                    low.remove(-out_num)  # Remove from max-heap
                    heapq.heapify(low)
                else:
                    high.remove(out_num)  # Remove from min-heap
                    heapq.heapify(high)
                rebalance()

        return result

# Example usage:
solution = Solution()

# Test case 1:
nums1 = [1,3,-1,-3,5,3,6,7]
k1 = 3
print(solution.medianSlidingWindow(nums1, k1))  # Output: [1.0, -1.0, -1.0, 3.0, 5.0, 6.0]

# Test case 2:
nums2 = [1,2,3,4,2,3,1,4,2]
k2 = 3
print(solution.medianSlidingWindow(nums2, k2))  # Output: [2.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0]




from collections import Counter

class Solution:
    def findSubstring(self, s: str, words: list[str]):
        if not s or not words or len(words[0]) == 0:
            return []

        word_len = len(words[0])
        word_count = len(words)
        total_len = word_len * word_count
        words_map = Counter(words)  # Frequency map of the words
        
        result = []
        
        # Sliding window approach
        for i in range(word_len):
            left = i  # left pointer of the sliding window
            right = i  # right pointer of the sliding window
            window_map = Counter()  # Map for the current window
            
            while right + word_len <= len(s):
                word = s[right:right + word_len]  # Get the current word from the right
                right += word_len  # Move the right pointer by word_len
                
                # If the word is in the words list, we add it to the window_map
                if word in words_map:
                    window_map[word] += 1
                    
                    # If the window contains more occurrences of the word than in words_map
                    # we need to shrink the window from the left
                    while window_map[word] > words_map[word]:
                        left_word = s[left:left + word_len]
                        window_map[left_word] -= 1
                        left += word_len  # Shrink the window from the left
                    
                    # If the window contains exactly the right count of each word, it's a valid substring
                    if right - left == total_len:
                        result.append(left)
                else:
                    # If the word is not in the list of words, reset the window
                    window_map.clear()
                    left = right = right  # Reset the pointers
        
        return result

# Example usage:
solution = Solution()

# Test case 1:
s1 = "barfoothefoobarman"
words1 = ["foo","bar"]
print(solution.findSubstring(s1, words1))  # Output: [0, 9


import heapq

class Solution:
    def smallestRange(self, nums):
        # Step 1: Initialize a min-heap and find the initial max element
        min_heap = []
        current_max = float('-inf')
        
        # Initialize the heap with the first element of each list
        for i in range(len(nums)):
            heapq.heappush(min_heap, (nums[i][0], i, 0))  # (value, list index, index in the list)
            current_max = max(current_max, nums[i][0])   # Track the current maximum
        
        # Step 2: Initialize the range
        smallest_range = float('inf')
        range_start, range_end = 0, 0
        
        # Step 3: Start sliding window approach
        while True:
            current_min, list_idx, element_idx = heapq.heappop(min_heap)  # Get the minimum element
            
            # Update the smallest range
            if current_max - current_min < smallest_range:
                smallest_range = current_max - current_min
                range_start, range_end = current_min, current_max
            
            # Step 4: If the current list is exhausted, break the loop
            if element_idx + 1 < len(nums[list_idx]):
                next_element = nums[list_idx][element_idx + 1]
                heapq.heappush(min_heap, (next_element, list_idx, element_idx + 1))
                current_max = max(current_max, next_element)  # Update the max if necessary
            else:
                break  # If any list is exhausted, we stop
        
        # Return the smallest range found
        return [range_start, range_end]

# Example usage:
solution = Solution()

# Test case 1:
nums1 = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
print(solution.smallestRange(nums1))  # Output: [1, 3]

# Test case 2:
nums2 = [[1, 10, 20], [2, 15, 30], [5, 25, 40]]
print(solution.smallestRange(nums2))  # Output: [5, 10]


class Solution:
    def findLength(self, nums1, nums2):
        # If either of the arrays is empty, return 0
        if not nums1 or not nums2:
            return 0
        
        # Initialize a 2D DP table with only two rows (space optimization)
        previous = [0] * (len(nums2) + 1)
        current = [0] * (len(nums2) + 1)
        max_len = 0
        
        # Iterate through each element of both arrays
        for i in range(1, len(nums1) + 1):
            for j in range(1, len(nums2) + 1):
                # If elements match, extend the length of the subarray
                if nums1[i - 1] == nums2[j - 1]:
                    current[j] = previous[j - 1] + 1
                    max_len = max(max_len, current[j])
                else:
                    current[j] = 0  # No match, reset the length
            previous = current[:]  # Move the current row to the previous row
        
        return max_len

# Example usage:
solution = Solution()

# Test case 1:
nums1 = [1, 2, 3, 2, 1]
nums2 = [3, 2, 1, 4, 7]
print(solution.findLength(nums1, nums2))  # Output: 3 (The subarray [3, 2, 1])

# Test case 2:
nums1 = [0, 0, 0, 0, 1]
nums2 = [1, 0, 0, 0, 0]
print(solution.findLength(nums1, nums2))  # Output: 4 (The subarray [0, 0, 0, 0])



# error
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        # Initialize a dummy node and a pointer to build the result linked list
        dummy_head = ListNode(0)
        current = dummy_head
        carry = 0
        
        # Traverse through both linked lists
        while l1 or l2 or carry:
            # Get the values of the current nodes of l1 and l2, or 0 if the node is None
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0
            
            # Calculate the sum and the new carry
            total = val1 + val2 + carry
            carry = total // 10  # Carry is either 0 or 1
            current.next = ListNode(total % 10)  # Create a new node with the result digit
            current = current.next  # Move the current pointer to the new node
            
            # Move to the next nodes in l1 and l2 if available
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
        
        # Return the head of the result list, which is the next of dummy_head
        return dummy_head.next

# Helper function to create a linked list from a list of integers
def create_linked_list(arr):
    dummy_head = ListNode(0)
    current = dummy_head
    for num in arr:
        current.next = ListNode(num)
        current = current.next
    return dummy_head.next

# Helper function to print a linked list
def print_linked_list(l):
    result = []
    while l:
        result.append(str(l.val))
        l = l.next
    print(" -> ".join(result))

# Example usage:
solution = Solution()

# Test case 1:
l1 = create_linked_list([2, 4, 3])  # Represents the number 342
l2 = create_linked_list([5, 6, 4])  # Represents the number 465
result = solution.addTwoNumbers(l1, l2)
print_linked_list(result)  # Output: 7 -> 0 -> 8, which represents 807



# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        # Edge case: if the list is empty or has only one node, no rotation needed
        if not head or not head.next or k == 0:
            return head
        
        # Step 1: Find the length of the list and the last node
        length = 1
        tail = head
        while tail.next:
            tail = tail.next
            length += 1
        
        # Step 2: Find the effective rotation steps
        k = k % length  # In case k is larger than the length of the list
        if k == 0:
            return head  # No need to rotate if k is a multiple of the length
        
        # Step 3: Make the list circular by connecting the tail to the head
        tail.next = head
        
        # Step 4: Find the new tail, which is at position length - k - 1
        new_tail = head
        for _ in range(length - k - 1):
            new_tail = new_tail.next
        
        # Step 5: The new head is the next node of the new tail
        new_head = new_tail.next
        # Break the circular list by setting the new tail's next to None
        new_tail.next = None
        
        return new_head

# Helper function to create a linked list from a list of integers
def create_linked_list(arr):
    dummy_head = ListNode(0)
    current = dummy_head
    for num in arr:
        current.next = ListNode(num)
        current = current.next
    return dummy_head.next

# Helper function to print a linked list
def print_linked_list(l):
    while l:
        print(l.val, end=" -> " if l.next else "\n")
        l = l.next

# Example usage:
solution = Solution()

# Test case 1: Rotate a list by 2 places
head = create_linked_list([1, 2, 3, 4, 5])
result = solution.rotateRight(head, 2)
print_linked_list(result)  # Expected output: 4 -> 5 -> 1 -> 2 -> 3

# Test case 2: Rotate a list by 3 places
head = create_linked_list([1, 2, 3, 4, 5])
result = solution.rotateRight(head, 3)
print_linked_list(result)  # Expected output: 3 -> 4 -> 5 -> 1 -> 2


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        # Create a dummy node which will be used to simplify edge cases
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        current = head
        
        while current:
            # Check if the current node is a duplicate
            if current.next and current.val == current.next.val:
                # Skip all nodes with the same value as current
                while current.next and current.val == current.next.val:
                    current = current.next
                # Link the previous distinct node to the next distinct node
                prev.next = current.next
            else:
                # Move prev pointer to current if no duplicate
                prev = prev.next
            
            # Move current pointer ahead
            current = current.next
        
        return dummy.next

# Helper function to create a linked list from a list of integers
def create_linked_list(arr):
    dummy_head = ListNode(0)
    current = dummy_head
    for num in arr:
        current.next = ListNode(num)
        current = current.next
    return dummy_head.next

# Helper function to print a linked list
def print_linked_list(l):
    while l:
        print(l.val, end=" -> " if l.next else "\n")
        l = l.next

# Example usage:
solution = Solution()

# Test case 1: List with duplicates
head = create_linked_list([1, 2, 2, 3, 3, 4, 5])
result = solution.deleteDuplicates(head)
print_linked_list(result)  # Expected output: 1 -> 4 -> 5

# Test case 2: List with no duplicates
head = create_linked_list([1, 2, 3, 4, 5])
result = solution.deleteDuplicates(head)
print_linked_list(result)  # Expected output: 1 -> 2 -> 3 -> 4 -> 5

# Test case 3: List with all duplicates
head = create_linked_list([1, 1, 1, 1, 1])
result = solution.deleteDuplicates(head)
print_linked_list(result)  # Expected output: (empty list)


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        # Edge case: if the list is empty or contains only one node
        if not head or not head.next:
            return head
        
        # Dummy node that will make it easier to handle the head
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy  # `prev` is the node before the pair to be swapped
        
        while prev.next and prev.next.next:
            # Identify the two nodes to be swapped
            first = prev.next
            second = first.next
            
            # Swap the two nodes
            first.next = second.next
            second.next = first
            prev.next = second  # Connect previous node to the second node
            
            # Move `prev` two steps forward for the next pair
            prev = first
        
        return dummy.next

# Helper function to create a linked list from a list of integers
def create_linked_list(arr):
    dummy_head = ListNode(0)
    current = dummy_head
    for num in arr:
        current.next = ListNode(num)
        current = current.next
    return dummy_head.next

# Helper function to print a linked list
def print_linked_list(l):
    while l:
        print(l.val, end=" -> " if l.next else "\n")
        l = l.next

# Example usage:
solution = Solution()

# Test case 1: Swap every two adjacent nodes
head = create_linked_list([1, 2, 3, 4])
result = solution.swapPairs(head)
print_linked_list(result)  # Expected output: 2 -> 1 -> 4 -> 3

# Test case 2: List with an odd number of nodes
head = create_linked_list([1, 2, 3])
result = solution.swapPairs(head)
print_linked_list(result)  # Expected output: 2 -> 1 -> 3

# Test case 3: List with only one node
head = create_linked_list([1])
result = solution.swapPairs(head)
print_linked_list(result)  # Expected output: 1 (no swap)

class Solution(object):

    def __init__(self):
        self.visited = {}

    def cloneGraph(self, node):
        """
        :type node: Node
        :rtype: Node
        """
        if not node:
            return node

        # 如果该节点已经被访问过了，则直接从哈希表中取出对应的克隆节点返回
        if node in self.visited:
            return self.visited[node]

        # 克隆节点，注意到为了深拷贝我们不会克隆它的邻居的列表
        clone_node = Node(node.val, [])

        # 哈希表存储
        self.visited[node] = clone_node

        # 遍历该节点的邻居并更新克隆节点的邻居列表
        if node.neighbors:
            clone_node.neighbors = [self.cloneGraph(n) for n in node.neighbors]

        return clone_node


