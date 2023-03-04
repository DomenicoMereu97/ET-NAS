# ET-NAS
## Introduction
This paper presents ET-NAS, a novel algorithm that leverages training-free measures and evolutionary search to reduce computational demands and search time in neural architecture search (NAS). Our work introduces the LossSlope metric, which exhibits higher correlation with final test accuracy than previously proposed metrics. Additionally, we conduct an ablation study to explore the effectiveness of different combinations of training-free measures.

## Methodology
We began by analyzing two existing NAS algorithms: random search using NASWOT score and REA. We implemented the NASWOT score as described in the original paper (Mellor, J., Turner, J., Storkey, A., Crowley, E. J. (2021, July). Neural architecture search without training) and calculated it for each network in the topology search space of NATS-Bench using three datasets: Cifar-10, Cifar-100, and ImageNet16-120.

To develop ET-NAS, we utilized a population-based evolutionary algorithm that iteratively generates new neural network architectures, evaluates them using training-free metrics, and selects the best-performing architectures for further mutation. We compute the training-free metric, LogSynFlow (Cavagnero, N., Robbiano, L., Caputo, B., Averta, G. (2022). FreeREA: Training-Free Evolution-based Architecture Search), for each architecture in the population, and select the best-performing architecture as the parent for mutation.

In each iteration, we mutate the parent architecture to produce a child architecture, and evaluate its performance using training-free metrics. We select the best-performing child architectures for the next iteration. To refine the search space further, we compute the NASWOT score for each architecture in the selected set, and use it to compute the Pareto fronts of the architecture space. Finally, we compute our metric LossSlope for each architecture in the Pareto fronts, and select the architecture closest to the utopian point with coordinates [1,1,1].

![Screenshot_4](https://user-images.githubusercontent.com/75221419/222916858-acb83dda-ff66-455d-8c66-8fde1d153617.jpg)

## Kendall and Spearman correlation for the training-free measures

![correlation](https://user-images.githubusercontent.com/75221419/222916922-772b8ed0-422c-4023-a003-24434eae5929.jpg)

## Results

![results](https://user-images.githubusercontent.com/75221419/222917093-180f93de-b670-4231-b7ac-148a8d9e0da7.jpg)


## Conclusion
Our proposed ET-NAS algorithm significantly reduces computational demands and search time in NAS by combining training-free measures and evolutionary search. We introduce the LossSlope metric, which exhibits higher correlation with final test accuracy than previously proposed metrics. Our ablation study demonstrates that different combinations of training-free measures can improve performance in NAS.

Major Implementation Details can be found in the report.

MLDL course @ Polito project for Academic Year 21/22
