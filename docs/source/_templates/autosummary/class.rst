{% set no_inherited_members = [
   'mdcraft.analysis.base.Hash',
   'mdcraft.analysis.reader.LAMMPSDumpTrajectoryReader'
] %}

{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   {% if fullname not in no_inherited_members -%}
   :inherited-members:
   {% endif -%}

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :nosignatures:
   {% for item in methods %}
      {%- if not item.startswith('_') and (
         fullname not in no_inherited_members or item not in inherited_members
      ) %}
         ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}