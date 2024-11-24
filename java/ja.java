import java.util.PriorityQueue;
import java.util.Arrays;

public class Solution {
    public int findMaximizedCapital(int k, int w, int[] profits, int[] capital) {
        int n = profits.length;

        // Combine projects into an array of [capital, profit]
        int[][] projects = new int[n][2];
        for (int i = 0; i < n; i++) {
            projects[i][0] = capital[i];
            projects[i][1] = profits[i];
        }

        // Sort projects by capital requirement
        Arrays.sort(projects, (a, b) -> Integer.compare(a[0], b[0]));

        // Max-heap to store profits of projects that can be started
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>((a, b) -> Integer.compare(b, a));

        int i = 0;

        while (k > 0) {
            // Add all projects that can be started with current capital to the heap
            while (i < n && projects[i][0] <= w) {
                maxHeap.add(projects[i][1]);
                i++;
            }

            // If no projects can be started, break
            if (maxHeap.isEmpty()) {
                break;
            }

            // Pick the most profitable project
            w += maxHeap.poll();
            k--;
        }

        return w;
    }
}
