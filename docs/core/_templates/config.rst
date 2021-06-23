{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:

   {% block methods %}

   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
   {% for item in all_methods %}
      {%- if item not in inherited_members and not item.startswith('_') or item in ['__init__', '__call__'] %}
         ~{{ name }}.{{ item }}
      {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in attributes %}
      {%- if not item.startswith('_') and not item in ['beta',
                                                       'capacity',
                                                       'activation',
                                                       'address_model_config',
                                                       'batch_norm',
                                                       'batch_size',
                                                       'categorical_config',
                                                       'categorical_threshold_log_multiplier',
                                                       'categorical_weight',
                                                       'check_frequency',
                                                       'checkpoint_path',
                                                       'clip_gradients',
                                                       'continuous_config',
                                                       'continuous_weight',
                                                       'custom_stop_metric',
                                                       'decay_rate',
                                                       'decay_steps',
                                                       'decomposed_continuous_config',
                                                       'dict_cache_size',
                                                       'distribution',
                                                       'engine_config',
                                                       'good_enough_metric',
                                                       'high_freq_weight',
                                                       'identifier_config',
                                                       'increase_batch_size_every',
                                                       'initial_boost',
                                                       'latent_size',
                                                       'learning_manager_config',
                                                       'learning_rate',
                                                       'low_freq_weight',
                                                       'max_batch_size',
                                                       'max_to_keep',
                                                       'max_training_time',
                                                       'min_num_unique',
                                                       'model_builder_config',
                                                       'moving_average',
                                                       'must_reach_metric',
                                                       'n_checks_no_improvement',
                                                       'nan_config',
                                                       'nan_weight',
                                                       'network',
                                                       'num_layers',
                                                       'optimizer',
                                                       'patience',
                                                       'person_model_config',
                                                       'postcode_model_config',
                                                       'residual_depths',
                                                       'sample_size',
                                                       'stop_metric_name',
                                                       'synthesis_batch_size',
                                                       'temperature',
                                                       'tol',
                                                       'use_checkpointing',
                                                       'use_engine_loss',
                                                       'value_factory_config',
                                                       'weight_decay',
                                                       'gender_female_regex',
                                                       'gender_male_regex',
                                                       'gender_non_binary_regex',
                                                       'genders',
                                                       'title_female_regex',
                                                       'title_male_regex',
                                                       'title_non_binary_regex'] %}
         ~{{ name }}.{{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}