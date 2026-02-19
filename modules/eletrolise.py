#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eletrolise.py
Módulo para modelagem de eletrolisadores para produção de hidrogênio verde
Baseado nas equações do artigo (Seção 2.1)

Referências:
- Eq. 2.1 a 2.8: Fundamentos de eletrólise
- Eq. 2.9: Produção de hidrogênio
- Eq. 2.26: Relação com corrente de Faraday
- Eq. 2.29: Tensão de operação
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ParametrosEletrolisador:
    """
    Parâmetros específicos para cada tecnologia de eletrolisador
    Baseado na Tabela 2.1 do artigo
    """
    eficiencia: float          # Eficiência energética (η)
    tensao_reversivel: float   # Tensão reversível (V_rev) em V
    coeficiente_transferencia: float  # Coeficiente de transferência (α)
    densidade_corrente_troca: float   # Densidade de corrente de troca (j0) em A/m²
    resistencia_ohmica: float  # Resistência ôhmica específica (R_ohm) em Ω·m²
    temperatura_operacao: float  # Temperatura de operação (°C)
    pressao_operacao: float    # Pressão de operação (bar)
    vida_util_anos: int        # Vida útil estimada
    custo_capex_usd_kw: float  # Custo de capital (USD/kW)


class Eletrolisador:
    """
    Classe principal para modelagem de eletrolisadores
    
    Suporta três tecnologias:
    - AEL: Eletrólise Alcalina
    - PEMEL: Eletrólise com Membrana de Troca de Prótons
    - SOEL: Eletrólise de Óxido Sólido
    
    Implementa todas as equações da Seção 2.1 do artigo
    """
    
    # Constantes físicas (SI)
    R = 8.314  # Constante dos gases (J/mol·K)
    F = 96485  # Constante de Faraday (C/mol)
    h_L = 33.33  # Poder calorífico inferior do H2 (kWh/kg) - LHV
    h_HV = 39.39  # Poder calorífico superior do H2 (kWh/kg) - HHV
    massa_molar_h2 = 2.016  # g/mol
    
    # Parâmetros padrão para cada tecnologia (baseado no artigo)
    PARAMETROS_PADRAO = {
        'AEL': {
            'eficiencia': 0.68,
            'tensao_reversivel': 1.23,
            'coeficiente_transferencia': 0.5,
            'densidade_corrente_troca': 1e-3,
            'resistencia_ohmica': 0.001,
            'temperatura_operacao': 70,
            'pressao_operacao': 30,
            'vida_util_anos': 20,
            'custo_capex_usd_kw': 800
        },
        'PEMEL': {
            'eficiencia': 0.78,
            'tensao_reversivel': 1.23,
            'coeficiente_transferencia': 0.5,
            'densidade_corrente_troca': 1e-4,
            'resistencia_ohmica': 0.0008,
            'temperatura_operacao': 60,
            'pressao_operacao': 35,
            'vida_util_anos': 15,
            'custo_capex_usd_kw': 1200
        },
        'SOEL': {
            'eficiencia': 0.89,
            'tensao_reversivel': 0.95,
            'coeficiente_transferencia': 0.7,
            'densidade_corrente_troca': 1e-2,
            'resistencia_ohmica': 0.002,
            'temperatura_operacao': 750,
            'pressao_operacao': 1,
            'vida_util_anos': 10,
            'custo_capex_usd_kw': 2000
        }
    }
    
    def __init__(self, 
                 tipo: str = 'AEL',
                 potencia_nominal: float = 1000,
                 parametros_personalizados: Optional[Dict] = None):
        """
        Inicializa um eletrolisador
        
        Args:
            tipo: 'AEL', 'PEMEL', ou 'SOEL'
            potencia_nominal: Potência nominal em kW
            parametros_personalizados: Dicionário com parâmetros customizados
        
        Raises:
            ValueError: Se o tipo for inválido
        """
        self.tipo = tipo.upper()
        if self.tipo not in self.PARAMETROS_PADRAO:
            raise ValueError(f"Tipo inválido: {tipo}. Escolha entre {list(self.PARAMETROS_PADRAO.keys())}")
        
        self.P_nom = float(potencia_nominal)
        self.P_min = self.P_nom * 0.2  # Carga mínima típica: 20%
        
        # Carregar parâmetros
        self.parametros = self._carregar_parametros(parametros_personalizados)
        
        # Estado operacional
        self.potencia_atual = 0.0
        self.temperatura_atual = self.parametros.temperatura_operacao
        self.horas_operacao = 0
        self.producao_acumulada_kg = 0.0
        
        # Histórico para debug/análise
        self.historico = {
            'potencia': [],
            'producao': [],
            'temperatura': [],
            'tensao': []
        }
        
        logger.info(f"Eletrolisador {self.tipo} criado: {self.P_nom} kW")
    
    def _carregar_parametros(self, personalizados: Optional[Dict]) -> ParametrosEletrolisador:
        """
        Carrega parâmetros da tecnologia escolhida
        
        Args:
            personalizados: Parâmetros customizados (opcional)
        
        Returns:
            ParametrosEletrolisador configurado
        """
        base = self.PARAMETROS_PADRAO[self.tipo].copy()
        
        if personalizados:
            base.update(personalizados)
        
        return ParametrosEletrolisador(
            eficiencia=base['eficiencia'],
            tensao_reversivel=base['tensao_reversivel'],
            coeficiente_transferencia=base['coeficiente_transferencia'],
            densidade_corrente_troca=base['densidade_corrente_troca'],
            resistencia_ohmica=base['resistencia_ohmica'],
            temperatura_operacao=base['temperatura_operacao'],
            pressao_operacao=base['pressao_operacao'],
            vida_util_anos=base['vida_util_anos'],
            custo_capex_usd_kw=base['custo_capex_usd_kw']
        )
    
    # ==================== EQUAÇÕES PRINCIPAIS ====================
    
    def calcular_producao(self, P_in: float) -> float:
        """
        Calcula a produção de hidrogênio
        
        Eq. 2.9 do artigo:
        m_el = (P_in * η_el) / h_L
        
        Args:
            P_in: Potência elétrica de entrada (kW)
        
        Returns:
            Produção de H2 (kg/h)
        """
        if P_in <= 0:
            return 0.0
        
        # Limitar à potência nominal se necessário
        P_efetiva = min(P_in, self.P_nom)
        
        # Eq. 2.9
        producao = (P_efetiva * self.parametros.eficiencia) / self.h_L
        
        return producao
    
    def calcular_producao_por_corrente(self, I_dc: float, eficiencia_faraday: float = 0.95) -> float:
        """
        Calcula produção de H2 usando a Lei de Faraday
        
        Eq. 2.26 do artigo:
        f_H2 = (η_f * I_dc) / F
        
        Args:
            I_dc: Corrente contínua (A)
            eficiencia_faraday: Eficiência de Faraday (η_f)
        
        Returns:
            Produção molar (mol/s) e mássica (kg/h)
        """
        # Produção molar (mol/s) - Eq. 2.26
        producao_molar = (eficiencia_faraday * I_dc) / self.F
        
        # Converter para kg/h
        producao_kg_h = producao_molar * (self.massa_molar_h2 / 1000) * 3600
        
        return {
            'mol_s': producao_molar,
            'kg_h': producao_kg_h
        }
    
    def calcular_tensao_operacao(self, j: float, T: Optional[float] = None) -> float:
        """
        Calcula a tensão de operação da célula
        
        Eq. 2.29 do artigo:
        V = V_rev + V_act + V_ohm
        
        Args:
            j: Densidade de corrente (A/m²)
            T: Temperatura (K) - opcional
        
        Returns:
            Tensão da célula (V)
        """
        if T is None:
            T = self.temperatura_atual + 273.15  # Converter para Kelvin
        
        # Tensão reversível - Eq. 2.29
        V_rev = self.parametros.tensao_reversivel
        
        # Sobretensão de ativação - Eq. 2.8
        V_act = self.calcular_sobretencao_ativacao(j, T)
        
        # Sobretensão ôhmica
        V_ohm = j * self.parametros.resistencia_ohmica
        
        V_total = V_rev + V_act + V_ohm
        
        return V_total
    
    def calcular_sobretencao_ativacao(self, j: float, T: Optional[float] = None) -> float:
        """
        Calcula a sobretensão de ativação usando equação de Tafel
        
        Eq. 2.8 do artigo:
        η_H2 = (2.3RT/αF) * log(j/j0)
        
        Para HER (Reação de Evolução de Hidrogênio)
        
        Args:
            j: Densidade de corrente (A/m²)
            T: Temperatura (K)
        
        Returns:
            Sobretensão de ativação (V)
        """
        if T is None:
            T = self.temperatura_atual + 273.15
        
        # Evitar log de zero ou negativo
        if j <= 0:
            return 0.0
        
        # Garantir que j/j0 > 0
        razao_corrente = max(j / self.parametros.densidade_corrente_troca, 1e-10)
        
        # Eq. 2.8 - Tafel equation
        termo = (2.3 * self.R * T) / (self.parametros.coeficiente_transferencia * self.F)
        eta = termo * np.log10(razao_corrente)
        
        return max(eta, 0)  # Sobretensão não negativa em operação normal
    
    def calcular_potencia_por_corrente(self, I_dc: float, V_celula: float, n_celulas: int = 100) -> float:
        """
        Calcula potência total a partir da corrente
        
        Args:
            I_dc: Corrente (A)
            V_celula: Tensão por célula (V)
            n_celulas: Número de células em série
        
        Returns:
            Potência total (kW)
        """
        P = I_dc * V_celula * n_celulas / 1000  # Converter para kW
        return P
    
    # ==================== MÉTODOS OPERACIONAIS ====================
    
    def operar(self, potencia_solicitada: float, delta_t_horas: float = 1.0) -> Dict:
        """
        Opera o eletrolisador por um período de tempo
        
        Args:
            potencia_solicitada: Potência solicitada (kW)
            delta_t_horas: Intervalo de tempo (horas)
        
        Returns:
            Dicionário com resultados da operação
        """
        # Verificar limites operacionais
        if potencia_solicitada < self.P_min and potencia_solicitada > 0:
            logger.warning(f"Potência {potencia_solicitada:.1f} kW abaixo do mínimo ({self.P_min:.1f} kW)")
            # Pode optar por desligar ou operar no mínimo
        
        if potencia_solicitada > self.P_nom:
            logger.warning(f"Potência {potencia_solicitada:.1f} kW acima do nominal ({self.P_nom:.1f} kW)")
            potencia_efetiva = self.P_nom
        else:
            potencia_efetiva = max(0, potencia_solicitada)
        
        # Calcular produção
        producao_kg_h = self.calcular_producao(potencia_efetiva)
        producao_periodo = producao_kg_h * delta_t_horas
        
        # Estimar tensão (simplificado)
        # Assumindo densidade de corrente proporcional à potência
        area_celula = 100  # m² (valor típico)
        j = (potencia_efetiva * 1000) / (self.parametros.tensao_reversivel * area_celula * 100)  # A/m²
        V = self.calcular_tensao_operacao(j)
        
        # Atualizar estado
        self.potencia_atual = potencia_efetiva
        self.horas_operacao += delta_t_horas
        self.producao_acumulada_kg += producao_periodo
        
        # Registrar histórico
        self.historico['potencia'].append(potencia_efetiva)
        self.historico['producao'].append(producao_periodo)
        self.historico['temperatura'].append(self.temperatura_atual)
        self.historico['tensao'].append(V)
        
        resultado = {
            'potencia_operacao': potencia_efetiva,
            'producao_kg_h': producao_kg_h,
            'producao_periodo_kg': producao_periodo,
            'tensao_celula': V,
            'temperatura': self.temperatura_atual,
            'eficiencia_instantanea': self.calcular_eficiencia_instantanea(potencia_efetiva)
        }
        
        return resultado
    
    def calcular_eficiencia_instantanea(self, potencia: float) -> float:
        """
        Calcula eficiência instantânea considerando carga parcial
        
        A eficiência varia com a carga (curva típica)
        
        Args:
            potencia: Potência de operação (kW)
        
        Returns:
            Eficiência instantânea (0-1)
        """
        if potencia <= 0:
            return 0.0
        
        # Carga relativa
        carga_rel = potencia / self.P_nom
        
        # Modelo simplificado de eficiência vs carga
        # Máxima eficiência em carga nominal, menor em carga parcial
        if carga_rel < 0.3:
            fator_carga = 0.85  # 15% menos eficiente
        elif carga_rel < 0.6:
            fator_carga = 0.92
        elif carga_rel < 0.9:
            fator_carga = 0.98
        else:
            fator_carga = 1.0
        
        return self.parametros.eficiencia * fator_carga
    
    # ==================== ANÁLISES ECONÔMICAS ====================
    
    def calcular_custo_capex(self, taxa_cambio_usd_brl: float = 5.0) -> float:
        """
        Calcula custo de capital (CAPEX) em Reais
        
        Args:
            taxa_cambio_usd_brl: Taxa de câmbio USD/BRL
        
        Returns:
            CAPEX em R$
        """
        custo_usd = self.P_nom * self.parametros.custo_capex_usd_kw
        return custo_usd * taxa_cambio_usd_brl
    
    def estimar_degradacao(self, anos_operacao: float) -> float:
        """
        Estima degradação da eficiência ao longo do tempo
        
        Args:
            anos_operacao: Anos de operação
        
        Returns:
            Fator de degradação (1 = sem degradação)
        """
        # Modelo linear de degradação: ~0.5% ao ano
        taxa_degradacao_anual = 0.005
        fator = max(1.0 - taxa_degradacao_anual * anos_operacao, 0.7)
        return fator
    
    # ==================== MÉTODOS DE RELATÓRIO ====================
    
    def resumo(self) -> Dict:
        """
        Retorna um resumo do estado do eletrolisador
        """
        return {
            'tipo': self.tipo,
            'potencia_nominal_kw': self.P_nom,
            'eficiencia_nominal': self.parametros.eficiencia,
            'temperatura_operacao_c': self.parametros.temperatura_operacao,
            'horas_operacao': self.horas_operacao,
            'producao_acumulada_kg': self.producao_acumulada_kg,
            'potencia_atual_kw': self.potencia_atual,
            'custo_capex_brl': self.calcular_custo_capex(),
            'vida_util_anos': self.parametros.vida_util_anos
        }
    
    def get_historico(self) -> Dict:
        """
        Retorna o histórico de operação
        """
        return self.historico.copy()
    
    def reset_historico(self):
        """Reseta o histórico de operação"""
        self.historico = {
            'potencia': [],
            'producao': [],
            'temperatura': [],
            'tensao': []
        }
        logger.info("Histórico resetado")
    
    def __str__(self) -> str:
        """Representação em string do eletrolisador"""
        return (f"Eletrolisador {self.tipo} | "
                f"{self.P_nom:.0f} kW | "
                f"η={self.parametros.eficiencia*100:.1f}% | "
                f"Produção acumulada: {self.producao_acumulada_kg:.1f} kg H₂")


# ==================== EXEMPLO DE USO ====================

def exemplo_uso():
    """
    Função de exemplo demonstrando o uso do módulo
    """
    print("\n" + "="*60)
    print("🔬 EXEMPLO DE USO DO MÓDULO DE ELETRÓLISE")
    print("="*60)
    
    # Criar eletrolisadores de diferentes tipos
    ael = Eletrolisador('AEL', potencia_nominal=1000)
    pemel = Eletrolisador('PEMEL', potencia_nominal=800)
    soel = Eletrolisador('SOEL', potencia_nominal=500)
    
    print(f"\n📊 ELETROLISADORES CRIADOS:")
    print(f"   {ael}")
    print(f"   {pemel}")
    print(f"   {soel}")
    
    # Testar produção com diferentes potências
    print(f"\n⚡ PRODUÇÃO DE H₂ (kg/h):")
    for p in [200, 500, 800, 1000]:
        prod_ael = ael.calcular_producao(p)
        prod_pemel = pemel.calcular_producao(min(p, pemel.P_nom))
        print(f"   {p:4d} kW → AEL: {prod_ael:5.2f} | PEMEL: {prod_pemel:5.2f}")
    
    # Simular operação por um dia
    print(f"\n⏱️ SIMULAÇÃO DE OPERAÇÃO (24h):")
    ael.reset_historico()
    
    for hora in range(24):
        # Perfil típico: mais potência durante o dia
        if 8 <= hora <= 18:
            potencia = 800 + np.random.normal(0, 50)
        else:
            potencia = 300 + np.random.normal(0, 30)
        
        resultado = ael.operar(potencia)
        
        if hora % 6 == 0:  # Mostrar a cada 6 horas
            print(f"   Hora {hora:2d}: {resultado['potencia_operacao']:5.1f} kW → "
                  f"{resultado['producao_periodo_kg']:5.2f} kg H₂")
    
    print(f"\n📈 RESUMO DA SIMULAÇÃO:")
    print(f"   {ael}")
    print(f"   Produção total: {ael.producao_acumulada_kg:.1f} kg H₂")
    print(f"   Horas operação: {ael.horas_operacao:.0f} h")
    
    # Comparação de tensões
    print(f"\n🔋 TENSÃO DE OPERAÇÃO (j=1000 A/m²):")
    for elz in [ael, pemel, soel]:
        V = elz.calcular_tensao_operacao(1000)
        print(f"   {elz.tipo}: V={V:.3f} V (V_rev={elz.parametros.tensao_reversivel} V)")
    
    return ael, pemel, soel


if __name__ == "__main__":
    # Executar exemplo quando o módulo for executado diretamente
    exemplo_uso()
    
    print("\n" + "="*60)
    print("✅ Módulo de eletrólise carregado com sucesso!")
    print("="*60)