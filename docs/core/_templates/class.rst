{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:

   {% block methods %}

      {% if methods %}
         .. rubric:: Methods

         .. autosummary::
            :toctree: {{ objname }}
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
            :toctree: {{ objname }}
         {% for item in attributes %}
            ~{{ name }}.{{ item }}
         {%- endfor %}
      {% endif %}
   {% endblock %}
