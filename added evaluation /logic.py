from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection


# So, I have to use a strategy pattern (one of the design patterns) in my application. In the strategy pattern, there is business_logic(Common_logic) and concrete_strategy. (You can use this blog for building strategy pattern https://refactoring.guru/design-patterns/strategy/python/example)
# ASSUME that, each server is already running. We will implement more detail for random forest and SVM.

# WHEN YOU BUILD A STRATEGY PATTERN…

# 1.For business_logic(Common Logic) in strategy pattern, we will use this function as common logic.. df = pd.read_csv('../dataset/frauddetection.csv'). This is because commonly random forest and SVM have to bring datasets.

# 2. We will add Lasso regression. Lasso also uses the same dataset(frauddetection.csv). Made it as an API.

# 3. We will make Evaluations for this Machine learning application. Please be aware of..! we have to SHOW visualization and SAVE the result on local desktop (Don’t use database. Don’t use cloud storage. Just local saving is fine) You can use these blogs to build Evaluation. https://medium.com/@polanitzer/fraud-detection-in-python-predict-fraudulent-credit-card-transactions-73992335dd90
# HOWEVER, To call the evaluation, We also have to make APIs(Logistic Regression, Random Forest, and Support Vector Machine).
# Please remember.. all results and visualization results should be saved.

# 4. For random forest class(Concrete strategies), we will call preprocess -&gt; random forest functionality -&gt; Evaluation
# (When we call the random forest class, then the preprocess API will be called -&gt; Random Forest Algorithm API will be called -&gt; Evaluation API will be called)
# 5. For SVM class(Concrete strategies), We will call preprocess -&gt; SVM -&gt; Evaluation
# (When we call the SVM class, then the preprocess API will be called -&gt; SVM API will be called -&gt; Evaluation API will be called)

# 6. Draw Class Diagram for this application.

class Context():
    """
    The Context defines the interface of interest to clients.
    """

    def __init__(self, strategy: Strategy) -> None:
        """
        Usually, the Context accepts a strategy through the constructor, but
        also provides a setter to change it at runtime.
        """
        self._strategy = strategy

    @property
    def strategy(self) -> Strategy:
        """
        The Context maintains a reference to one of the Strategy objects. The
        Context does not know the concrete class of a strategy. It should work
        with all strategies via the Strategy interface.
        """

        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Strategy) -> None:
        """
        Usually, the Context allows replacing a Strategy object at runtime.
        """

        self._strategy = strategy

    def do_some_business_logic(self) -> None:
        """
        The Context delegates some work to the Strategy object instead of
        implementing multiple versions of the algorithm on its own.
        """
        # ...
        df = pd.read_csv('dataset/process_data.csv')
        print(df)
        y = df['fraud']
        X =  df.loc[:, df.columns != 'fraud']
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        print("Context: Sorting data using the strategy (not sure how it'll do it)")
        result = self._strategy.do_algorithm(["e", "b", "c", "d", "a"])
        print(",".join(result))

        # ...


class Strategy(ABC):
    """
    The Strategy interface declares operations common to all supported versions
    of some algorithm.

    The Context uses this interface to call the algorithm defined by Concrete
    Strategies.
    """

    @abstractmethod
    def do_algorithm(self, data: List):
        pass


"""
Concrete Strategies implement the algorithm while following the base Strategy
interface. The interface makes them interchangeable in the Context.
"""


class ConcreteStrategyA(Strategy):
    def do_algorithm(self, data: List) -> List:
        return sorted(data)


class ConcreteStrategyB(Strategy):
    def do_algorithm(self, data: List) -> List:
        return reversed(sorted(data))

class ConcreteStrategyC(Strategy):
    def do_algorithm(self, data: List) -> List:
        return data

if __name__ == "__main__":
    # The client code picks a concrete strategy and passes it to the context.
    # The client should be aware of the differences between strategies in order
    # to make the right choice.

    context = Context(ConcreteStrategyA())
    print("Client: Strategy is set to normal sorting.")
    context.do_some_business_logic()
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

    print("Client: Strategy is set to reverse sorting.")
    context.strategy = ConcreteStrategyB()
    context.do_some_business_logic()

    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

    print("Client: Strategy is set to Normal.")
    context.strategy = ConcreteStrategyC()
    context.do_some_business_logic()