# Masterclass - Creación de Agentes con Python

Este repositorio contiene el código que acompaña a la masterclass de creación de agentes con Python.

- [01 - Introducción](./01_intro.ipynb): Implementa un chatbot básico.
- [02 - Herramientas](./02_herramientas.ipynb): Mejoramos el chatbot con el uso de herramientas para convertirlo en un agente.
- [03 - Memoria](./03_memoria.ipynb): Implementamos la memoria en el agente para que recuerde conversaciones pasadas.
- [04 - HIL](./04_hil.ipynb): Implementamos *human-in-the-loop* en el agente para que pueda pedir inputs extra al usuario durante la conversación.
- [05 - Agente de investigación](./05_agente.ipynb): Aplicamos todo lo aprendido en el desarrollo de un agente de búsqueda para la validación de ideas de negocio.

## ¿Quieres aprender más?

- Tutoriales GRATIS en mi blog (https://www.juansensio.com/blog) y Youtube (https://www.youtube.com/@juansensio)
- Curso online: El Curso de IA (https://www.elcursodeia.com/)
    -   Introducción a la Computación con Python (GRATIS)
    -   Análisis de Datos con Python (GRATIS)
    -   Machine Learning con Scikit-Learn
    -   Deep Learning con Pytorch
    -   Y mucho más...
- Libro: Introducción a la Inteligencia Artificial con Python (https://savvily.es/libros/introduccion-a-la-inteligencia-artificial-con-python/)
- Únete a la comunidad de Discord (https://discord.gg/aTeEKXzKbs) 
- Sígueme en redes sociales (https://x.com/juansensio; https://linkedin.com/in/juanbpedro)


## Requisitos

Instala `uv` para tu sistema operativo siguiendo las instrucciones en https://docs.astral.sh/uv/getting-started/installation/.

Después, instala las dependencias del proyecto con:

```bash
uv sync
```

Si empiezas un nuevo proyecto desde cero, primero tendrás que crear el entorno virtual con:

```bash
uv init
```

Luego, podrás instalar las dependencias con `uv add <nombre_del_paquete>`, por ejemplo:

```bash
uv add langgraph
```