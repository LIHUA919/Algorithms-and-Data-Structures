class MaxHeap
    def initialize
      @heap = []
    end
  
    # Insert a value into the heap
    def push(value)
      @heap << value
      bubble_up(@heap.size - 1)
    end
  
    # Extract the maximum value from the heap
    def pop
      return nil if @heap.empty?
  
      max_value = @heap[0]
      @heap[0] = @heap.pop
      bubble_down(0) unless @heap.empty?
      max_value
    end
  
    # Check if the heap is empty
    def empty?
      @heap.empty?
    end
  
    private
  
    def bubble_up(index)
      return if index.zero?
  
      parent_index = (index - 1) / 2
      if @heap[index] > @heap[parent_index]
        @heap[index], @heap[parent_index] = @heap[parent_index], @heap[index]
        bubble_up(parent_index)
      end
    end
  
    def bubble_down(index)
      left_child = 2 * index + 1
      right_child = 2 * index + 2
      largest = index
  
      largest = left_child if left_child < @heap.size && @heap[left_child] > @heap[largest]
      largest = right_child if right_child < @heap.size && @heap[right_child] > @heap[largest]
  
      if largest != index
        @heap[index], @heap[largest] = @heap[largest], @heap[index]
        bubble_down(largest)
      end
    end
  end
  
  def find_maximized_capital(k, w, profits, capital)
    n = profits.size
    projects = []
  
    # Combine profits and capital into a list of projects
    n.times { |i| projects << [capital[i], profits[i]] }
  
    # Sort projects by capital requirement
    projects.sort_by! { |cap, _| cap }
  
    max_heap = MaxHeap.new
  
    i = 0
    while k > 0
      # Add all the projects that can be started with the current capital
      while i < n && projects[i][0] <= w
        max_heap.push(projects[i][1]) # push profit into the heap
        i += 1
      end
  
      # If there are no projects that can be started, break
      break if max_heap.empty?
  
      # Select the most profitable project
      w += max_heap.pop
  
      k -= 1
    end
  
    w
  end
  
  # Example usage
  k = 10
  w = 0
  profits = [1, 2, 3]
  capital = [0, 1, 2]
  
  puts find_maximized_capital(k, w, profits, capital)
  