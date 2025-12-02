"""Phase 1-A: Multi-Agent System Implementation.

This module implements a ReAct agent that uses RecipeSearchTool and 
SentimentAnalysisTool to provide intelligent menu recommendations.

Note: Requires 'langchain' package for agents. Install with: pip install langchain
"""
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import logging

logger = logging.getLogger("agent_system")

try:
    from langchain.agents import create_openai_tools_agent, AgentExecutor
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False
    logger.warning("[AGENT] langchain.agents not available. Install with: pip install langchain")



# -------------------- Agent Prompt Template --------------------

AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 전문 메뉴 추천 에이전트입니다.

사용 가능한 도구:
1. search_recipes: 레시피 벡터 검색 (사용자 요구사항에 맞는 메뉴 찾기)
2. get_user_sentiment: 감성 분석 (사용자 기분 파악)

추천 프로세스:
1. 먼저 get_user_sentiment로 사용자 감성 분석
2. search_recipes로 관련 레시피 검색 (top_k=3~5)
3. 감성과 검색 결과를 종합하여 최종 추천

규칙:
- 최대 2개 메뉴 추천
- 각 추천에는 이유와 매칭 요소 포함
- 감성이 부정적이면 위로가 되는 메뉴 선택
- 감성이 긍정적이면 활기찬 메뉴 제안
- 검색 결과가 없으면 일반적인 안전 메뉴 제안
- 불필요한 도구 호출 자제 (최대 3회 반복)

현재 시간: {current_time}
사용자 프로필: {user_profile}
"""),
    ("human", "{user_input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


# -------------------- Agent Creation --------------------

def create_recommendation_agent(
    tools: List,
    llm: ChatOpenAI | None = None,
    max_iterations: int = 3,
    max_execution_time: float = 30.0
):
    """Create a ReAct agent for menu recommendations.
    
    Args:
        tools: List of LangChain tools (RecipeSearchTool, SentimentAnalysisTool).
        llm: ChatOpenAI instance (default: gpt-4o with temperature=0.7).
        max_iterations: Maximum agent iterations to prevent infinite loops (default: 3).
        max_execution_time: Maximum execution time in seconds (default: 30.0).
    
    Returns:
        AgentExecutor with tool binding or None if agents not available.
    """
    if not AGENTS_AVAILABLE:
        logger.error("[AGENT] Cannot create agent: langchain.agents not installed")
        return None
    
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    # Create agent with tools
    agent = create_openai_tools_agent(
        llm=llm,
        tools=tools,
        prompt=AGENT_PROMPT
    )
    
    # Wrap in executor with safety limits
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time,
        verbose=True,  # Enable logging for debugging
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )
    
    logger.info(f"[AGENT] Created AgentExecutor with {len(tools)} tools, max_iterations={max_iterations}")
    return agent_executor


# -------------------- Agent Execution --------------------

def run_agent_recommendation(
    agent_executor,
    user_input: str,
    current_time: str = "",
    user_profile: str = ""
) -> Dict[str, Any]:
    """Run agent to generate recommendations.
    
    Args:
        agent_executor: AgentExecutor instance or None.
        user_input: User query.
        current_time: Current time string (optional).
        user_profile: User profile summary (optional).
    
    Returns:
        Dictionary with recommendations and metadata.
    """
    if not AGENTS_AVAILABLE or agent_executor is None:
        logger.error("[AGENT] Cannot run: agent_executor is None or agents not available")
        return {
            "recommendations": [{
                "title": "Agent 시스템 사용 불가",
                "reason": "langchain 패키지가 설치되지 않았습니다. 'pip install langchain'으로 설치해주세요.",
                "match_factors": ["설정 오류"]
            }],
            "error": "langchain.agents not installed",
            "mode": "Multi-Agent System (Error)"
        }
    
    logger.info(f"[AGENT] Starting recommendation for: {user_input[:50]}...")
    
    try:
        # Execute agent
        result = agent_executor.invoke({
            "user_input": user_input,
            "current_time": current_time or "정보 없음",
            "user_profile": user_profile or "설정 안됨"
        })
        
        output = result.get("output", "")
        intermediate_steps = result.get("intermediate_steps", [])
        
        logger.info(f"[AGENT] Completed with {len(intermediate_steps)} tool calls")
        
        # Parse agent output into structured format
        recommendations = parse_agent_output(output)
        
        return {
            "recommendations": recommendations,
            "raw_output": output,
            "tool_calls": len(intermediate_steps),
            "intermediate_steps": intermediate_steps,
            "mode": "Multi-Agent System"
        }
    
    except Exception as e:
        logger.error(f"[AGENT] Execution failed: {e}", exc_info=True)
        return {
            "recommendations": [],
            "error": str(e),
            "mode": "Multi-Agent System (Error)"
        }


# -------------------- Output Parsing --------------------

def parse_agent_output(output: str) -> List[Dict[str, Any]]:
    """Parse agent's text output into structured recommendations.
    
    Args:
        output: Agent's text response.
    
    Returns:
        List of recommendation dictionaries.
    """
    # Simple parsing - look for numbered recommendations
    recommendations = []
    lines = output.split('\n')
    
    current_rec = None
    for line in lines:
        line = line.strip()
        
        # Detect recommendation start (numbered list)
        if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
            if current_rec:
                recommendations.append(current_rec)
            
            # Extract title (remove numbering)
            title = line.lstrip('0123456789.-•) ').strip()
            current_rec = {
                "title": title[:50] if title else "추천 메뉴",
                "reason": "",
                "match_factors": ["Agent 추천"]
            }
        elif current_rec and line:
            # Add to reason
            current_rec["reason"] += " " + line
    
    # Add last recommendation
    if current_rec:
        recommendations.append(current_rec)
    
    # Clean up reasons
    for rec in recommendations:
        rec["reason"] = rec["reason"].strip()[:200]  # Limit length
    
    # If no structured recommendations found, create one from full output
    if not recommendations and output:
        recommendations.append({
            "title": "Agent 추천",
            "reason": output[:200],
            "match_factors": ["Agent 분석"]
        })
    
    logger.info(f"[AGENT] Parsed {len(recommendations)} recommendations from output")
    return recommendations[:2]  # Limit to 2 recommendations
