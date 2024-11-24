

(define (search-insert nums target)
  (define (binary-search left right)
    (if (<= left right)
        (let* ([mid (+ left (quotient (- right left) 2))]
               [mid-value (list-ref nums mid)])
          (cond
            [(= mid-value target) mid] ; Target found
            [(< mid-value target) (binary-search (+ mid 1) right)] ; Search right
            [else (binary-search left (- mid 1))])) ; Search left
        left)) ; Target not found, return insertion index
  (binary-search 0 (sub1 (length nums))))






(define (search-matrix matrix target)
  (let* ([m (length matrix)]
         [n (length (first matrix))]
         [left 0]
         [right (- (* m n) 1)])
    (let loop ([left left] [right right])
      (if (> left right)
          #f ; Target not found
          (let* ([mid (+ left (quotient (- right left) 2))]
                 [row (quotient mid n)]
                 [col (remainder mid n)]
                 [mid-value (list-ref (list-ref matrix row) col)])
            (cond
              [(= mid-value target) #t] ; Target found
              [(< mid-value target) (loop (+ mid 1) right)] ; Search right
              [else (loop left (- mid 1))])))))) ; Search left




#lang racket

(define (find-peak-element nums)
  (define (binary-search left right)
    (if (>= left right)
        left
        (let* ([mid (quotient (+ left right) 2)]
               [mid-val (list-ref nums mid)]
               [next-val (list-ref nums (+ mid 1))])
          (if (< mid-val next-val)
              (binary-search (+ mid 1) right)
              (binary-search left mid)))))
  (binary-search 0 (- (length nums) 1)))


#lang racket

(define (search nums target)
  (define (binary-search left right)
    (if (> left right)
        -1 ; Target not found
        (let* ([mid (+ left (quotient (- right left) 2))]
               [mid-value (list-ref nums mid)])
          (cond
            [(= mid-value target) mid] ; Found the target
            [(<= (list-ref nums left) mid-value) ; Left half is sorted
             (if (and (>= target (list-ref nums left)) (< target mid-value))
                 (binary-search left (- mid 1)) ; Search in left half
                 (binary-search (+ mid 1) right))] ; Search in right half
            [else ; Right half is sorted
             (if (and (> target mid-value) (<= target (list-ref nums right)))
                 (binary-search (+ mid 1) right) ; Search in right half
                 (binary-search left (- mid 1)))])))) ; Search in left half
  (binary-search 0 (- (length nums) 1)))




#lang racket

(define (search-range nums target)
  ;; Helper function for binary search to find the first or last occurrence
  (define (binary-search left right find-first)
    (if (> left right)
        -1
        (let* ([mid (+ left (quotient (- right left) 2))]
               [mid-value (list-ref nums mid)])
          (cond
            [(= mid-value target)
             (if find-first
                 (let ([prev (binary-search left (- mid 1) find-first)])
                   (if (= prev -1) mid prev)) ; If no earlier occurrence, return mid
                 (let ([next (binary-search (+ mid 1) right find-first)])
                   (if (= next -1) mid next)))] ; If no later occurrence, return mid
            [(< mid-value target)
             (binary-search (+ mid 1) right find-first)] ; Search right
            [else
             (binary-search left (- mid 1) find-first)])))) ; Search left

  (let ([start (binary-search 0 (- (length nums) 1) #t)])
    (if (= start -1)
        '(-1 -1) ; Target not found
        (list start (binary-search 0 (- (length nums) 1) #f))))) ; Find last occurrence



#lang racket

(define (find-min nums)
  (define (binary-search left right)
    (if (= left right)
        (list-ref nums left) ; Found the minimum
        (let* ([mid (+ left (quotient (- right left) 2))]
               [mid-value (list-ref nums mid)]
               [right-value (list-ref nums right)])
          (if (> mid-value right-value)
              (binary-search (+ mid 1) right) ; Minimum in right half
              (binary-search left mid)))))    ; Minimum in left half
  (binary-search 0 (- (length nums) 1)))



#lang racket

(define (find-median-sorted-arrays nums1 nums2)
  ;; Ensure nums1 is the smaller array
  (if (> (length nums1) (length nums2))
      (find-median-sorted-arrays nums2 nums1)
      (let* ([m (length nums1)]
             [n (length nums2)]
             [half-len (quotient (+ m n +1) 2)])
        ;; Helper function for binary search
        (define (binary-search left right)
          (if (> left right)
              #f ; Base case, should not be reached
              (let* ([i (quotient (+ left right) 2)]
                     [j (- half-len i)]
                     [max-left1 (if (= i 0) -inf.0 (list-ref nums1 (- i 1)))]
                     [min-right1 (if (= i m) +inf.0 (list-ref nums1 i))]
                     [max-left2 (if (= j 0) -inf.0 (list-ref nums2 (- j 1)))]
                     [min-right2 (if (= j n) +inf.0 (list-ref nums2 j))])
                ;; Check partition validity
                (if (and (<= max-left1 min-right2) (<= max-left2 min-right1))
                    ;; Calculate median
                    (if (odd? (+ m n))
                        (max max-left1 max-left2) ; Odd case
                        (/ (+ (max max-left1 max-left2) 
                              (min min-right1 min-right2)) 2)) ; Even case
                    ;; Adjust partitions
                    (if (> max-left1 min-right2)
                        (binary-search left (- i 1)) ; Move left
                        (binary-search (+ i 1) right)))))) ; Move right
        ;; Start binary search
        (binary-search 0 m))))






;; overtime
#lang racket

(define (find-kth-largest nums k)
  ;; Convert kth largest to kth smallest (0-indexed)
  (define k-index (- (length nums) k))

  ;; Helper function: partition the array around a pivot
  (define (partition left right pivot-index)
    (let* ([pivot (list-ref nums pivot-index)]
           [nums-swap (lambda (i j)
                        (let ([temp (list-ref nums i)])
                          (set! nums (list-set nums i (list-ref nums j)))
                          (set! nums (list-set nums j temp))))])
      (nums-swap pivot-index right) ; Move pivot to end
      (let ([store-index left])
        (for ([i (in-range left right)])
          (when (< (list-ref nums i) pivot)
            (nums-swap i store-index)
            (set! store-index (+ store-index 1))))
        (nums-swap store-index right) ; Move pivot to its final place
        store-index)))

  ;; Helper function: quickselect
  (define (quickselect left right)
    (if (= left right)
        (list-ref nums left) ; Base case: only one element
        (let ([pivot-index (quotient (+ left right) 2)]) ; Choose middle element as pivot
          (let ([new-pivot-index (partition left right pivot-index)])
            (cond
              [(= new-pivot-index k-index) (list-ref nums new-pivot-index)] ; Found the k-th smallest
              [(< new-pivot-index k-index) (quickselect (+ new-pivot-index 1) right)] ; Search right
              [else (quickselect left (- new-pivot-index 1))]))))) ; Search left

  ;; Start quickselect


  (quickselect 0 (- (length nums) 1)))







  #lang racket

;; Max-Heap helper functions
(define (heap-add heap value)
  (let ([heap (append heap (list value))])
    (heapify-up heap (- (length heap) 1))))

(define (heapify-up heap index)
  (if (<= index 0)
      heap
      (let* ([parent (quotient (- index 1) 2)]
             [heap (if (> (list-ref heap index) (list-ref heap parent))
                       (swap heap index parent)
                       heap)])
        (heapify-up heap parent))))

(define (heap-remove heap)
  (if (empty? heap)
      (error "Heap underflow")
      (let ([last (list-ref heap (- (length heap) 1))])
        (if (= (length heap) 1)
            (values (list) last)
            (let ([heap (cons last (rest heap))])
              (values (heapify-down (take heap (- (length heap) 1)) 0)
                      (first heap)))))))

(define (heapify-down heap index)
  (let ([left-child (+ (* 2 index) 1)]
        [right-child (+ (* 2 index) 2)]
        [largest index])
    (when (< left-child (length heap))
      (set! largest (if (> (list-ref heap left-child) (list-ref heap largest))
                        left-child
                        largest)))
    (when (< right-child (length heap))
      (set! largest (if (> (list-ref heap right-child) (list-ref heap largest))
                        right-child
                        largest)))
    (if (= largest index)
        heap
        (heapify-down (swap heap index largest) largest))))

(define (swap lst i j)
  (let ([temp (list-ref lst i)])
    (list-set (list-set lst i (list-ref lst j)) j temp)))

;; Main function to maximize capital
(define (maximize-capital k w profits capital)
  ;; Combine projects as (capital, profit) pairs and sort by capital
  (define projects (sort (map list capital profits) #:key first #:order <))

  ;; Max-Heap for profits
  (define profit-heap '())
  (define idx 0)

  ;; Iterate up to k times to select projects
  (for ([i (in-range k)])
    ;; Add all projects that can be started with current capital
    (let loop ()
      (when (and (< idx (length projects))
                 (<= (first (list-ref projects idx)) w))
        (set! profit-heap (heap-add profit-heap (second (list-ref projects idx))))
        (set! idx (+ idx 1))
        (loop)))

    ;; If there are no projects in the heap, stop
    (when (empty? profit-heap)
      (return w))

    ;; Pick the most profitable project and update capital
    (define-values (profit-heap max-profit) (heap-remove profit-heap))
    (set! w (+ w max-profit)))

  ;; Return the final capital
  w)





  ;;Line 68: Char 8: return: unbound identifier
  ;;in: return
  ;;compilation context...:
   ;;solution.rkt
  ;;location...:
   ;;prog_joined.rkt:80:7

