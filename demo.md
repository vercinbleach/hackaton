# Demo y evaluación

## Objetivo

El training y la demo son dos piezas distintas.

El training produce una variante de GLiNER2 ajustada con LoRA. La demo es un eval interactivo que compara cuatro sistemas sobre el mismo conjunto de consultas. Su objetivo es medir si el LoRA mejora la precisión sin perder la ventaja de velocidad y coste del modelo base.

## Sistemas comparados

| Sistema | Entrada | Salida |
| --- | --- | --- |
| Modelo SOTA | Lenguaje natural y skill de Cala | Plan estructurado |
| Cala interno | Lenguaje natural directo | Resultado Cala |
| GLiNER2 base | Lenguaje natural y schema | Plan estructurado |
| GLiNER2 LoRA | Lenguaje natural y el mismo schema | Plan estructurado |

Los tres sistemas que generan un plan comparten exactamente:

- El schema de salida.
- El validador.
- El compilador de Cala QL.
- El endpoint `/knowledge/query`.

Cala interno recibe la consulta en lenguaje natural porque esa es la capacidad que queremos usar como baseline. No se le obliga a pasar por nuestro parser.

Esta separación permite atribuir los fallos. Si el resultado final es incorrecto, podremos distinguir entre un plan mal generado, una compilación incorrecta o una interpretación inesperada de Cala.

## Dataset congelado

El eval usará entre 100 y 200 consultas que no aparecerán en train. Una vez revisadas, se congelan y no se modifican para favorecer ningún modelo.

El conjunto debe cubrir:

- Entidades concretas.
- Colecciones.
- Relaciones.
- Filtros numéricos.
- Combinaciones de filtros.
- Proyecciones de campos.
- Orden y límite.
- Consultas fuera de alcance.
- Consultas en español e inglés.

Cada caso tendrá una consulta, un plan gold revisado manualmente y, cuando corresponda, las propiedades esperadas del resultado.

```json
{
  "query": "Dame nombre y fundador de empresas creadas por exempleados de Google",
  "gold_plan": {
    "operation": "knowledge_query",
    "root": "companies",
    "filters": [
      {
        "path": ["founded", "previous_job"],
        "operator": "=",
        "value": "Google"
      }
    ],
    "return": ["name", "founder"]
  }
}
```

La partición mínima será:

- `train`: ejemplos usados para ajustar el LoRA.
- `validation`: ejemplos usados durante el desarrollo y la selección del checkpoint.
- `test`: benchmark congelado, sin exposición durante el entrenamiento.

## Eval del parser

La primera evaluación termina en el plan estructurado y no llama a Cala.

```text
consulta -> modelo -> plan -> validador -> métricas
```

Mediremos:

- Exact match del plan completo.
- Exact match de filtros.
- F1 de los campos solicitados en `return`.
- Precisión y recall de `unsupported`.
- Porcentaje de planes válidos según el schema.
- Latencia del modelo.
- Tokens y coste del modelo SOTA.

Esta evaluación compara directamente SOTA, GLiNER2 base y GLiNER2 LoRA. Cala interno queda fuera porque no expone el mismo contrato de plan.

## Eval end-to-end

La segunda evaluación ejecuta el flujo completo.

```text
consulta -> modelo -> plan -> Cala QL -> Cala -> resultado
```

Para Cala interno, el flujo será:

```text
consulta -> Cala natural -> resultado
```

Mediremos:

- Porcentaje de consultas Cala válidas.
- Corrección del resultado frente al gold.
- Columnas solicitadas y columnas devueltas.
- Campos extra no solicitados.
- Bytes y tokens de respuesta.
- Latencia total.
- Coste total.
- Porcentaje aceptado por FastPath.

Las métricas del parser y las de Cala se guardarán por separado. Un resultado incorrecto no contará automáticamente como fallo del modelo si el plan era correcto.

## Interfaz

El usuario puede elegir una consulta del test o escribir una consulta libre. La vista principal muestra cuatro columnas:

```text
SOTA | Cala natural | GLiNER2 base | GLiNER2 LoRA
```

Cada columna muestra:

- Plan, cuando el sistema lo produzca.
- Cala QL, cuando exista compilación.
- Resultado.
- Columnas devueltas.
- Latencia.
- Tokens y coste disponibles.
- Estado correcto, incorrecto o no evaluable.
- Punto de fallo, si lo hay.

La consulta libre sirve para explorar. No modifica el benchmark ni sus métricas oficiales.

## Lectura esperada de la demo

La demo debe permitir comprobar estas hipótesis, no darlas por ciertas antes de medir:

- Cala natural entiende la intención, pero puede devolver información adicional.
- El modelo SOTA logra buena precisión, con más latencia y coste.
- GLiNER2 base es rápido, pero falla en casos que requieren más composición.
- GLiNER2 LoRA conserva la velocidad del modelo base y mejora su precisión.

El resultado principal será una tabla agregada sobre el test congelado. La comparación interactiva permitirá abrir cada caso y entender por qué acertó o falló cada sistema.

## Resultado del hackathon

El benchmark reproducible es el producto técnico. La interfaz lo hace visible y permite inspeccionar casos concretos.

La entrega incluye:

- Dataset versionado con splits de train, validation y test.
- Planes gold revisados.
- Runner común para los cuatro sistemas.
- Validador y compilador compartidos.
- Métricas por caso y agregadas.
- Interfaz para comparar resultados.
- Configuración necesaria para repetir el eval.
