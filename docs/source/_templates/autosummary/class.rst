{% set no_inheritance_classes = [
   'mdcraft.analysis.base.Hash',
   'mdcraft.analysis.reader.LAMMPSDumpTrajectoryReader'
] %}

{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   {% if fullname not in no_inheritance_classes -%}
   :inherited-members:
   {% endif -%}

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :nosignatures:
   {% for item in methods %}
      {%- if not item.startswith('_') and (
         fullname not in no_inheritance_classes or item not in inherited_members
      ) %}
         ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}