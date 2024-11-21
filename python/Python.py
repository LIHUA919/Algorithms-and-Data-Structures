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