/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     struct TreeNode *left;
 *     struct TreeNode *right;
 * };
 */
/**
 * Note: The returned array must be malloced, assume caller calls free().
 */
double* averageOfLevels(struct TreeNode* root, int* returnSize) {
    if (!root) {
        *returnSize = 0;
        return NULL;
    }
    
    // Allocate maximum possible size for result
    double* result = (double*)malloc(10000 * sizeof(double));
    *returnSize = 0;
    
    // Create queue using array
    struct TreeNode** queue = (struct TreeNode**)malloc(10000 * sizeof(struct TreeNode*));
    int front = 0, rear = 0;
    
    // Enqueue root
    queue[rear++] = root;
    
    while (front < rear) {
        int levelSize = rear - front;
        double levelSum = 0;
        
        for (int i = 0; i < levelSize; i++) {
            struct TreeNode* node = queue[front++];
            levelSum += node->val;
            
            if (node->left) queue[rear++] = node->left;
            if (node->right) queue[rear++] = node->right;
        }
        
        result[*returnSize] = levelSum / levelSize;
        (*returnSize)++;
    }
    
    free(queue);
    return result;
}


/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     struct TreeNode *left;
 *     struct TreeNode *right;
 * };
 */
/**
 * Return an array of arrays of size *returnSize.
 * The sizes of the arrays are returned as *returnColumnSizes array.
 * Note: Both returned array and *columnSizes array must be malloced, assume caller calls free().
 */
int** levelOrder(struct TreeNode* root, int* returnSize, int** returnColumnSizes) {
    if (!root) {
        *returnSize = 0;
        return NULL;
    }
    
    // Initialize result arrays
    int** result = (int**)malloc(10000 * sizeof(int*));
    *returnColumnSizes = (int*)malloc(10000 * sizeof(int));
    *returnSize = 0;
    
    // Create queue
    struct TreeNode** queue = (struct TreeNode**)malloc(10000 * sizeof(struct TreeNode*));
    int front = 0, rear = 0;
    
    // Enqueue root
    queue[rear++] = root;
    
    while (front < rear) {
        int levelSize = rear - front;
        
        // Allocate space for current level
        result[*returnSize] = (int*)malloc(levelSize * sizeof(int));
        (*returnColumnSizes)[*returnSize] = levelSize;
        
        for (int i = 0; i < levelSize; i++) {
            struct TreeNode* node = queue[front++];
            result[*returnSize][i] = node->val;
            
            if (node->left) queue[rear++] = node->left;
            if (node->right) queue[rear++] = node->right;
        }
        
        (*returnSize)++;
    }
    
    free(queue);
    return result;
}


/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     struct TreeNode *left;
 *     struct TreeNode *right;
 * };
 */

/**
 * Return an array of arrays of size *returnSize.
 * The sizes of the arrays are returned as *returnColumnSizes array.
 * Note: Both returned array and *columnSizes array must be malloced, assume caller calls free().
 */
int** zigzagLevelOrder(struct TreeNode* root, int* returnSize, int** returnColumnSizes) {
    if (!root) {
        *returnSize = 0;
        *returnColumnSizes = NULL;
        return NULL;
    }
    
    // Initialize arrays to store results
    int** result = (int**)malloc(sizeof(int*) * 10000);
    *returnColumnSizes = (int*)malloc(sizeof(int) * 10000);
    *returnSize = 0;
    
    // Create queue for BFS
    struct TreeNode** queue = (struct TreeNode**)malloc(sizeof(struct TreeNode*) * 10000);
    int front = 0, rear = 0;
    
    // Add root to queue
    queue[rear++] = root;
    bool leftToRight = true;
    
    while (front < rear) {
        int levelSize = rear - front;
        (*returnColumnSizes)[*returnSize] = levelSize;
        
        // Allocate memory for current level
        result[*returnSize] = (int*)malloc(sizeof(int) * levelSize);
        
        for (int i = 0; i < levelSize; i++) {
            struct TreeNode* node = queue[front++];
            
            // Fill current level based on direction
            int index = leftToRight ? i : levelSize - 1 - i;
            result[*returnSize][index] = node->val;
            
            if (node->left) queue[rear++] = node->left;
            if (node->right) queue[rear++] = node->right;
        }
        
        (*returnSize)++;
        leftToRight = !leftToRight;
    }
    
    free(queue);
    return result;
}

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        if (!root) return {};
        
        vector<vector<int>> result;
        queue<TreeNode*> q;
        q.push(root);
        bool leftToRight = true;
        
        while (!q.empty()) {
            int levelSize = q.size();
            vector<int> currentLevel(levelSize);
            
            for (int i = 0; i < levelSize; i++) {
                TreeNode* node = q.front();
                q.pop();
                
                // Fill current level based on direction
                int index = leftToRight ? i : levelSize - 1 - i;
                currentLevel[index] = node->val;
                
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
            
            result.push_back(currentLevel);
            leftToRight = !leftToRight;
        }
        
        return result;
    }
};