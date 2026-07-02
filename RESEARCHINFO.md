this is my proposal for research with keepa maharjan
Mid-Execution Sorting Adaptation via Machine Learning
Keepa Maharjan
Advisor: Prof. Vladislav D. Veksler 


Introduction 
Sorting is one of the most fundamental operations in computer science that almost every system, from database engines to operating system schedulers, relies on [1]. The selection of an algorithm directly affects how fast or slow the system will run, and selecting the wrong one, especially on large or complex data, may increase the runtime significantly [2]. Traditional algorithms excel in specific conditions but cannot adapt when input characteristics change mid-execution. Hybrid approaches like Timsort [3] and Introsort [4] improved upon this by switching strategies using hand-crafted heuristics. At the same time, more recent work has applied machine learning to predict the optimal algorithm from static input features before execution begins [5].
Both approaches share a critical limitation: the switching decision is made before runtime behavior can be observed. Handcrafted heuristics cannot generalize beyond the conditions for which they were designed, and ML pre-selection commits to a choice before a single comparison has been made. More detailed information, such as comparison rates, progress speed, and the nature of the remaining array, which could be used for switching logic, will only be available after sorting is underway, yet to our knowledge, no existing system uses this information to reconsider its choice.
This research asks: in which cases does changing sorting algorithms during execution give a positive net performance, considering the switching costs, and can a machine learning model trained on features extracted mid-execution be able to accurately predict that decision?
Proposed Methods 
1. Algorithm Implementation and Instrumentation
QuickSort, MergeSort, and Insertion Sort will be implemented from scratch, each selected for its contrasting tradeoffs in speed, memory, and input sensitivity. The performance of each algorithm will be measured by the following attributes: number of comparisons, number of swaps, elapsed time, and the sorting progress at a fixed 50% execution checkpoint. For feasibility, the checkpoint will be implemented as a controlled pause after a predetermined proportion of algorithm steps. This checkpoint-based design keeps the framework tractable while providing a meaningful mid-execution window for observation and decision-making.

2. Dataset Construction and Input Categories
Algorithms will be benchmarked across seven input categories: random, sorted, reverse sorted, nearly sorted, duplicate-heavy, mixed-pattern, and a real-world numeric dataset. With such variety, the framework will be exposed to the full range of conditions under which algorithm behavior varies significantly across implementations, and the resulting labeled dataset captures a broad and representative range of switching scenarios relevant to practical sorting contexts.





3. Switching Cost Analysis
At the 50% checkpoint, two paths will be evaluated: continuing with the current algorithm or transferring execution to an alternative starting from the partially sorted array state. The performance difference between these choices will be recorded in a Switching Cost Matrix, which measures the gain or loss associated with each possible transition under different input conditions. This matrix serves both as an analytical tool for understanding algorithm behavior and as the primary ground-truth source for model labeling.

4. Feature Extraction and Dataset Labeling
For every checkpoint instance, a feature vector will be constructed from measurable runtime signals such as comparison count, swap count, elapsed time, progress rate, and input category, will form a structured feature set. Each observation will be labeled with whether switching is beneficial, which algorithm is optimal, and the expected magnitude of gain or loss. This labeled dataset will form the foundation for all subsequent supervised model training and evaluation.

5. Predictive Model Development and Comparison
A Decision Tree and a Random Forest will be trained on the labeled dataset. Both models will predict switching decisions from mid-execution features alone and will be evaluated using accuracy, precision, recall, and F1-score, with additional attention paid to false positives and false negatives to understand the practical cost of mispredictions in a sorting context.

6. Adaptive System Prototype and Baseline Comparison
A prototype system will initiate sorting by selecting one initial algorithm, pause at the 50% checkpoint, extract the current runtime feature snapshot, and use the best-performing trained model to decide whether to continue or switch the sorting algorithm. It will be benchmarked against three baselines: always continuing with the original algorithm, TimSort, and IntroSort, to assess whether adaptive mid-execution switching yields measurable and consistent performance gains across varied input conditions.


Significance of Findings and Future Work 
This research is significant because it explores a new layer of adaptivity in sorting: decision-making during execution rather than only before it. While hybrid sorting and pre-execution algorithm selection are established ideas, switching guided by mid-execution machine learning is still a largely untouched research area. This work investigates whether runtime signals contain useful information to make better sorting decisions mid-execution, thus contributing to the broader field of self-adaptive algorithms and intelligent runtime optimization. Even when switching proves suboptimal, the framework yields valuable insight into when switching helps, when it harms, and which features are most informative.
This research paper also lays down the groundwork for the next series of work: experimenting with multiple checkpoints at 25%, 50%, and 75%, introducing more sorting algorithms, and designing more advanced state-transfer mechanisms to reduce switching overhead. In this regard, it is not only a short-term study but also a promising starting point for deeper exploration of runtime-adaptive algorithmic systems.
References 
[1] Liu, P. (2024). An in-depth study of sorting algorithms. Applied and Computational Engineering. https://doi.org/10.54254/2755-2721/92/20241750 
[2] Kaur, P., Mahajan, S., & Kour, H. (2017). A comparative study of various types of sorting techniques. International Journal of Advanced Research in Computer Science, 8(7), 1–5. https://ijarcs.info/index.php/Ijarcs/article/view/4239
[3] Auger, N., Jugé, V., Nicaud, C., & Pivoteau, C. (2018). On the Worst-Case Complexity of TimSort. arXiv preprint. https://arxiv.org/abs/1805.08612 
[4] Lammich, P. (2020). Efficient Verified Implementation of Introsort and Pdqsort. Automated Reasoning (IJCAR 2020), 12167, 307–323. https://pmc.ncbi.nlm.nih.gov/articles/PMC7324064/ 
[5] Majumdar, S., Jain, I., Kukreja, K., & Bhowmick, K. (2016). Adaptive Sorting Using Machine Learning. International Journal of Computer Science and Information Technologies, Vol. 7 (2), 490-493. https://www.ijcsit.com/docs/Volume%207/vol7issue2/ijcsit2016070209.pdf 


















