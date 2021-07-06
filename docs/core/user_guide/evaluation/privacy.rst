.. _privacy_guide:


=======
Privacy
=======

Synthesized aims to synthesize a dataset such that the statistical properties of the original dataset are preserved while providing
ample protection against privacy attacks. Synthesized’s privacy module provides various ways to assess the robustness of synthesized
data against different types of attribute inference attack.

Attribute inference attack refers to the situation when an attacker adversary might deduce, with significant probability, the value of
a hidden sensitive attribute from the values of other attributes. In practice, the attacker will have full access to the synthetic data
and partial access to the original data. The attacker will train a model using synthetic data, and then use the
trained model to predict the unknown value of the sensitive attribute using the known attributes of the original data. Hence, it is
important and useful to assess the vulnerability of synthetic dataset against the risk of inference attacks so that the privacy and
confidentiality of the original data is preserved.

Synthesized provides two main classes to assess the attribute inference attack:

    * :class:`~synthesized.insight.metrics.privacy.AttributeInferenceAttackML`
    * :class:`~synthesized.insight.metrics.privacy.AttributeInferenceAttackCAP`

Machine Learning (ML) models or Correct Attribution Probability (CAP) models are fit to the synthetic data. The fitted model
is then used to compute the privacy score of a sensitive column of the original dataset using predictors in the original dataset.
Privacy scores are between 0 and 1; 0 means negligible privacy and 1 means absolute privacy.


Attribute Inference Attack using ML
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~synthesized.insight.metrics.privacy.AttributeInferenceAttackML` trains a machine learning model to predict the sensitive
attribute using the synthesized dataset. The fitted model is then used to predict the sensitive values in the original dataset.
Finally, a privacy score is calculated based on the true value and the predicted value of the sensitive column in the original dataset.

:class:`~synthesized.insight.metrics.privacy.AttributeInferenceAttackML` package can be imported as:

.. ipython:: python

    from synthesized.insight.metrics.privacy import AttributeInferenceAttackML

The following example shows how to use it step-by-step:

Firstly, initialize :class:`~synthesized.insight.metrics.privacy.AttributeInferenceAttackML` with the name of choice of model,
the list of the column names which are predictors and the name of the sensitive column.

.. ipython:: python
    :verbatim:

    predictors = ['RevolvingUtilizationOfUnsecuredLines', 'SeriousDlqin2yrs']
    sensitive_col = 'age'
    privacy_metrics = AttributeInferenceAttackML('Linear', sensitive_col, predictors)

Next, call the class object with the original dataset and the synthetic dataset to compute the privacy
score of the synthetic dataset.

.. ipython:: python
    :verbatim:

    privacy_score = privacy_metrics(orig_df=credit_df, synth_df=credit_df_synth)
    print(privacy_score)
    >>> 0.1158


Attribute Inference Attack using CAP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:class:`~synthesized.insight.metrics.privacy.AttributeInferenceAttackCAP` computes the privacy score using ``CAP (Correct Attribution Probability)`` model.
It is modeled as the probability that an attribution is correct. It differs from the ML approach because it doesn't depend on the choice of the ML model and
its training.

It will find all the rows in the synthetic dataset corresponding to each predictors key of the
original dataset and then fetch the list of the sensitive entries from these rows of the synthetic dataset.
The frequency of the correct sensitive entry of the original dataset in this list of sensitive entries
is used to compute the privacy score.

:class:`~synthesized.insight.metrics.privacy.AttributeInferenceAttackCAP` package can be used as:

.. ipython:: python

    from synthesized.insight.metrics.privacy import AttributeInferenceAttackCAP

Given below are the two ways to filter the rows in synthetic dataset corresponding to the predictors key of the
original dataset.

GeneralizedCAP
^^^^^^^^^^^^^^
``GeneralizedCAP`` finds all the rows in the synthetic dataset that match **exactly** to the predictors key of the original dataset

.. ipython:: python
    :verbatim:

    predictors = ['NumberOfTime30-59DaysPastDueNotWorse', 'age']
    sensitive_col = 'SeriousDlqin2yrs'
    privacy_metrics = AttributeInferenceAttackCAP('GeneralizedCAP', sensitive_col, predictors)
    privacy_score = privacy_metrics(orig_df=credit_df, synth_df=credit_df_synth)
    print(privacy_score)
    >>> 0.2398

DistanceCAP
^^^^^^^^^^^
``DistanceCAP`` finds all the rows in the synthetic dataset that are closest neighbours (in terms of Hamming distance) to the predictors key of the original dataset

.. ipython:: python
    :verbatim:

    predictors = ['RevolvingUtilizationOfUnsecuredLines', 'age']
    sensitive_col = 'SeriousDlqin2yrs'
    privacy_metrics = AttributeInferenceAttackCAP('DistanceCAP', sensitive_col, predictors)
    privacy_score = privacy_metrics(orig_df=credit_df, synth_df=credit_df_synth)
    print(privacy_score)
    >>> 0.2385

.. note::
    If the predictor columns name list is not provided as an argument during initialization of the above classes then all the columns,
    except the sensitive column, will be used as predictors.
