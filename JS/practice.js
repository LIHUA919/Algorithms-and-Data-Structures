/*时间复杂度和空间复杂度是计算机科学中评估算法效率的两个重要指标。让我来解释一下这两个概念：
时间复杂度
时间复杂度用来衡量算法运行所需的时间随输入规模增长的变化趋势。它不是具体的时间（如秒或毫秒），而是一个数学表达式，通常用大O符号表示。
例如：

O(1)：常数时间，不管输入多大，执行时间都相同（如数组访问特定索引）
O(log n)：对数时间，随着输入增加，执行时间增长缓慢（如二分查找）
O(n)：线性时间，执行时间与输入大小成正比（如遍历数组）
O(n²)：平方时间，执行时间与输入大小的平方成正比（如简单的嵌套循环）
O(2^n)：指数时间，执行时间随输入增加呈爆炸性增长（如递归求解斐波那契数列）

空间复杂度
空间复杂度用来衡量算法运行所需的额外存储空间随输入规模增长的变化趋势。它也使用大O符号表示。
例如：

O(1)：常数空间，不管输入多大，额外空间需求都相同（如只使用几个临时变量）
O(n)：线性空间，额外空间需求与输入大小成正比（如创建一个与输入等大的数组）
O(n²)：平方空间，额外空间需求与输入大小的平方成正比（如创建一个二维矩阵）

实际例子
以矩阵置零算法为例：

时间复杂度 O(m × n)：

需要遍历 m×n 个元素两次（一次标记，一次置零）
随着矩阵大小增加，处理时间线性增长


空间复杂度 O(m + n)：

需要额外创建两个数组来标记行和列
一个长度为 m 的数组和一个长度为 n 的数组
随着矩阵维度增加，所需额外空间也线性增长



理解这些概念对于选择合适的算法、优化代码性能非常重要，特别是在处理大规模数据时。 */
// 先会一种做法就可以

/*var setZeroes = function(matrix) {
    const m = matrix.length, n = matrix[0].length
    const row = new Array(m).fill(false), col = new Array(n).fill(false)

    for(let i=0; i<m; i++) {
        for(let j=0; j<n; j++) {
            if(matrix[i][j] === 0) {
                row[i] = col[j] = true
            }
        }
    }

    for(let i=0; i<m; i++) {
        for(let j=0; j<n; j++) {
            if(row[i] || col[j]) {
                matrix[i][j] = 0
            }
        }
    }

} */


    let s = '([])';
    for (let ch of s) {
      console.log(ch);
    }

   /* pairs = new Map([[')', '('], [']', '['], ['}', '{']]);

    pairs.has(')')   // true
    pairs.has('(')   // false（左括号不在pairs里）
    pairs.has(']')   // true
*/
    const pairs = new Map([
        [')', '('],
        [']', '['],
        ['}', '{']
    ]);
    
    console.log(pairs.get(')'));  // 输出 '('
    console.log(pairs.get(']'));  // 输出 '['
    console.log(pairs.get('}'));  // 输出 '{'


    // 用栈的方法
var isValid = function(s) {
    const n = s.length
    if(n % 2 === 1) {
        return false
    }

    const pairs = new Map([
        [')', '('],
        [']', '['],
        ['}', '{']
    ])

    stack = []
    for(let ch of s) {
        if(pairs.has(ch)) {
            if(!stack.length || stack[stack.length - 1] !== pairs.get(ch)) {
                return false
            } stack.pop()
            
        }
        else {
            stack.push  (ch)
        }

    }
    return !stack.length    
}
