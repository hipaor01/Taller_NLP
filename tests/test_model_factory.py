from __future__ import annotations

import unittest
from unittest.mock import patch

from langchain_core.rate_limiters import InMemoryRateLimiter

from taller_nlp.model_factory import crear_modelo_chat


class TestFabricaModelos(unittest.TestCase):
    @patch("taller_nlp.model_factory.init_chat_model")
    def test_convierte_segundos_a_milisegundos_para_openrouter(self, init) -> None:
        esperado = object()
        init.return_value = esperado

        resultado = crear_modelo_chat(
            "openrouter:google/gemini-3.8-flash",
            temperatura=0,
            timeout_s=60,
            max_tokens=500,
        )

        self.assertIs(resultado, esperado)
        init.assert_called_once_with(
            "openrouter:google/gemini-3.8-flash",
            temperature=0,
            timeout=60_000,
            max_retries=1,
            reasoning={"exclude": True},
            max_tokens=500,
        )

    @patch("taller_nlp.model_factory.init_chat_model")
    def test_conserva_segundos_para_otros_proveedores(self, init) -> None:
        crear_modelo_chat(
            "openai:gpt-5-mini",
            temperatura=0.2,
            timeout_s=45,
        )

        init.assert_called_once_with(
            "openai:gpt-5-mini",
            temperature=0.2,
            timeout=45,
        )

    @patch("taller_nlp.model_factory.init_chat_model")
    def test_inyecta_el_mismo_limitador_en_el_modelo(self, init) -> None:
        limitador = InMemoryRateLimiter(requests_per_second=1)
        crear_modelo_chat(
            "openrouter:modelo",
            temperatura=0,
            timeout_s=10,
            limitador=limitador,
        )
        self.assertIs(init.call_args.kwargs["rate_limiter"], limitador)


if __name__ == "__main__":
    unittest.main()
