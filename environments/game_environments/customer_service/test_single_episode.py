#!/usr/bin/env python3
"""
Simple script to test a single customer service episode
Usage: python test_single_episode.py
"""

import asyncio
import os
from environments.game_environments.customer_service.customer_service_env import CustomerServiceEnv

async def test_single_episode():
    """Run a single episode to check output quality"""
    print("🧪 Testing single customer service episode...")
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("   Run: export OPENAI_API_KEY='your-key-here'")
        return
    
    print("🔧 Using OpenAI for all LLMs (Agent, Environment & Judge)")
    
    try:
        # Initialize environment
        env_config, server_configs = CustomerServiceEnv.config_init()
        
        # Override to use OpenAI for Agent LLM too
        server_configs[0].model_name = "gpt-4o-mini"
        server_configs[0].base_url = "https://api.openai.com/v1"
        server_configs[0].api_key = os.getenv("OPENAI_API_KEY")
        
        # Override config for single episode testing
        env_config.eval_episodes = 1
        env_config.steps_per_eval = 1
        env_config.use_wandb = False  # Disable wandb for testing
        
        env = CustomerServiceEnv(
            config=env_config,
            server_configs=server_configs,
            slurm=False,
            testing=False  # Use real API, not mock
        )
        
        print("📋 Environment initialized")
        
        # Get a test item
        test_item = await env.get_eval_item()
        print(f"\n🎯 Test Scenario:")
        print(f"   Agent scenario: {test_item['agent_scenario']}")
        print(f"   Environment scenario: {test_item['env_scenario']}")
        print(f"   Seed: {test_item['seed']}")
        
        print(f"\n🚀 Starting episode...")
        
        # Run single trajectory
        result = await env.collect_trajectory(test_item, is_eval=False)
        
        if result and result[0]:
            scored_item = result[0]
            score = scored_item['scores']
            print(f"\n✅ Episode completed!")
            print(f"📊 Final score: {score:.3f}")
            print(f"🏆 Result: {'SUCCESS' if score > 0 else 'FAILED'}")
            
            if hasattr(env, 'log_file'):
                print(f"📝 Detailed log saved to: {env.log_file}")
        else:
            print(f"\n❌ Episode failed to complete")
            
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_single_episode()) 