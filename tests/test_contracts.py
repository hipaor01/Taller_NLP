from __future__ import annotations

import unittest

from pydantic import ValidationError

from taller_nlp import (
    ConfiguracionAgente,
    InformeEvaluacion,
    LlamadaHerramienta,
    RespuestaAgente,
    ResultadoPregunta,
)


def respuesta(**cambios: object) -> RespuestaAgente:
    datos = {
        "respuesta": "Respuesta",
        "fuente": "texto",
        "latencia_ms": 10,
    }
    datos.update(cambios)
    return RespuestaAgente(**datos)


class TestContratosRespuesta(unittest.TestCase):
    def test_llamada_exige_exactamente_resultado_o_error(self) -> None:
        base = {
            "id": "call-1",
            "nombre": "list_available",
            "duracion_ms": 0,
        }
        with self.assertRaises(ValidationError):
            LlamadaHerramienta(**base)
        with self.assertRaises(ValidationError):
            LlamadaHerramienta(**base, resultado="ok", error="fallo")

        llamada = LlamadaHerramienta(**base, resultado="ok")
        self.assertTrue(llamada.exitosa)

    def test_chunk_ids_solo_permitidos_en_search_filings(self) -> None:
        with self.assertRaisesRegex(ValidationError, "Solo search_filings"):
            LlamadaHerramienta(
                id="call-1",
                nombre="read_section",
                duracion_ms=1,
                chunk_ids=("ACME-2024-1A-0000",),
                resultado="ok",
            )

    def test_respuesta_vacia_solo_es_valida_si_hay_error(self) -> None:
        with self.assertRaises(ValidationError):
            respuesta(respuesta="")
        fallida = respuesta(respuesta="", fuente="ninguna", error="boom")
        self.assertFalse(fallida.exitosa)

    def test_tokens_totales_solo_si_existe_telemetria_completa(self) -> None:
        self.assertEqual(
            respuesta(tokens_entrada=7, tokens_salida=3).tokens_totales,
            10,
        )
        self.assertIsNone(respuesta(tokens_entrada=7).tokens_totales)

    def test_normaliza_cita_textual_y_rechaza_la_vacia(self) -> None:
        self.assertEqual(respuesta(cita="  texto citado  ").cita, "texto citado")
        with self.assertRaisesRegex(ValidationError, "cita"):
            respuesta(cita="   ")


class TestResultadosEInforme(unittest.TestCase):
    def test_acierto_exige_todos_los_criterios_de_su_familia(self) -> None:
        extractiva = ResultadoPregunta(
            id_pregunta="q1",
            familia="extractiva",
            respuesta_agente=respuesta(),
            cita_existe=True,
            cita_respalda=True,
            trayectoria_correcta=True,
            # recall@k se informa aparte y no condiciona los tres evaluadores.
            recall_at_k=0,
        )
        numerica = ResultadoPregunta(
            id_pregunta="q2",
            familia="numerica",
            respuesta_agente=respuesta(cifra=100, unidad="USD", fuente="xbrl"),
            cifra_correcta=False,
            trayectoria_correcta=True,
        )
        self.assertTrue(extractiva.acierto)
        self.assertFalse(numerica.acierto)

    def test_informe_agrega_metricas_y_coberturas(self) -> None:
        resultados = (
            ResultadoPregunta(
                id_pregunta="q1",
                familia="extractiva",
                respuesta_agente=respuesta(coste_usd=0.01),
                cita_existe=True,
                cita_respalda=True,
                trayectoria_correcta=True,
                recall_at_k=1,
            ),
            ResultadoPregunta(
                id_pregunta="q2",
                familia="numerica",
                respuesta_agente=respuesta(
                    cifra=100,
                    unidad="USD",
                    fuente="xbrl",
                    latencia_ms=30,
                ),
                cifra_correcta=False,
                trayectoria_correcta=True,
            ),
        )
        informe = InformeEvaluacion(
            nombre_agente="baseline",
            ruta_jsonl="golden.jsonl",
            resultados=resultados,
            k_retrieval=5,
            tolerancia_absoluta=1,
            tolerancia_relativa=1e-6,
        )
        self.assertEqual(informe.aciertos_totales, 1)
        self.assertEqual(informe.tasa_acierto_global, 0.5)
        self.assertEqual(informe.recall_at_k_medio, 1.0)
        self.assertEqual(informe.cobertura_recall, 0.5)
        self.assertEqual(informe.coste_medio_usd, 0.01)
        self.assertEqual(informe.cobertura_coste, 0.5)
        self.assertEqual(informe.latencia_media_ms, 20)

    def test_informe_rechaza_ids_repetidos(self) -> None:
        resultado = ResultadoPregunta(
            id_pregunta="q1",
            familia="numerica",
            respuesta_agente=respuesta(cifra=1, unidad="USD", fuente="xbrl"),
            cifra_correcta=True,
            trayectoria_correcta=True,
        )
        with self.assertRaisesRegex(ValidationError, "únicos"):
            InformeEvaluacion(
                nombre_agente="a",
                ruta_jsonl="g.jsonl",
                resultados=(resultado, resultado),
                k_retrieval=1,
                tolerancia_absoluta=0,
                tolerancia_relativa=0,
            )


class TestConfiguracion(unittest.TestCase):
    def test_exige_declarar_juntas_las_tarifas_de_tokens(self) -> None:
        with self.assertRaisesRegex(ValidationError, "tarifas"):
            ConfiguracionAgente(
                modelo="modelo",
                system_prompt="prompt",
                precio_entrada_usd_millon_tokens=0.5,
            )

    def test_normaliza_texto_y_acepta_limite_cero_por_tool(self) -> None:
        config = ConfiguracionAgente(
            modelo=" modelo ",
            system_prompt=" instrucciones ",
            max_llamadas_total=4,
            limites_por_herramienta={"read_section": 0},
        )
        self.assertEqual(config.modelo, "modelo")
        self.assertEqual(config.system_prompt, "instrucciones")

    def test_rechaza_limite_por_tool_superior_al_total(self) -> None:
        with self.assertRaisesRegex(ValidationError, "max_llamadas_total"):
            ConfiguracionAgente(
                modelo="modelo",
                system_prompt="prompt",
                max_llamadas_total=2,
                limites_por_herramienta={"search_filings": 3},
            )

    def test_rechaza_backoff_con_espera_inicial_superior_a_maxima(self) -> None:
        with self.assertRaisesRegex(ValidationError, "espera_inicial"):
            ConfiguracionAgente(
                modelo="modelo",
                system_prompt="prompt",
                espera_inicial_rate_limit_s=30,
                espera_maxima_rate_limit_s=20,
            )


if __name__ == "__main__":
    unittest.main()
