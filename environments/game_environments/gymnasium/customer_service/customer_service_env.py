import json
import logging
import random
from typing import Dict, List, Optional, Tuple

from atroposlib.envs.base import APIServerConfig, BaseEnv, BaseEnvConfig, ScoredDataItem
from atroposlib.type_definitions import Item, Message
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer
from atroposlib.utils.tool_call_parser import parse_tool_call

logger = logging.getLogger(__name__)

class CustomerServiceEnvConfig(BaseEnvConfig):
    """
    Configuration for the CustomerService environment.
    """

    env_name: str = "CustomerService"
    max_episode_turns: int = 20
    eval_episodes: int = 100 # TODO: experiment to adjust
    dataset_path: str = "/shared/share_mala/melody/atropos/environments/game_environments/gymnasium/customer_service/customer_service_dataset.json"
    env_first_message: str = "You are going to role-play as a customer experiencing a specific issue. \n\nYou must adhere to the following guidelines:\n1. Only respond to the agent's questions and provide relevant information when prompted. \n2.Your responses should be concise and reflect a typical, non-technical customer's knowledge level. \n3. You are going to know what the solution for the specific issue is, but do not give away this information at any point. Your goal is to observe if the customer service agent can ask relevant questions and arrive at the specific issue. \n4.When the agent asks you about a potential solution, you should determine what the outcome would be based on your knowledge about the true underlying problem, and let the agent know the result. For example, if your car is not starting because of lack of fuel, and the agent suggests you to check the fuel level in your car, it should fix your issue. Use your reasoning ability to decide whether the suggested solution works, given that the solution to your problem is: {env} \n5. Reply with 'Goal reached' (and nothing more) if the agent has reached or solved your particular issue. \n\nYour role-play starts now. The particular problem you, as the customer, are facing is: {agent}, and the solution to this problem is: {env}. Note that you should never give away or tell the agent what will solve your particular problem, your goal is to see if the agent can come up with the potential solution themselves."
    env_default_response: str = "Sorry, I do not know the answer to this question. Could you please help me in some other way?"
    judge_prompt_agent: str = "You are the judge of a customer service agent trying to help a customer in a particular scenario. You would be given the the agent's response, and you have to verify whether the agent has been able to solve the problem of the customer correctly. \n\nThe particular scenario the customer is facing is: {agent}, and the solution to their problem is: {env} \n\nReply with <VALID> if the agent is successful in suggesting the correct solution, otherwise reply with <NOTVALID>."
    env_optional_message: str = "\n\n(Remember the customer service agent is trying to solve your particular scenario. The solution for your particular scenario is {env}. If the agent's proposed solution does not fix your problem, let the agent know that it does not solve your problem. Use your reasoning ability to decide if the particular suggestion would work for your scenario. If the agent's proposed solution is correct or they have guessed the underlying problem correctly, reply with 'Goal reached'.)"
    judge_prompt_suffix: str = "\n\nNow judge whether the agent has been successful in making the correct suggestion to solve the customer's problem. Use your reasoning ability to decide whether the agent's response would solve the customer's problem, which is {env}. For example, if the customer's car is not starting because of a lack of fuel, and the agent suggests to check the fuel level in the car, it should fix the issue. Reply with <VALID> if they have been successful, otherwise reply with <NOTVALID>. \n\nAnswer:"
    agent_system_prompt: str = "You are going to role-play as a customer service agent and you have to help a customer resolve their issue. Your goal is to gather enough information to diagnose the problem and provide solution. \n\nYour instructions are the following: \n1.You will need to ask targeted questions or suggest particular actions to the customer to gather the necessary details. \n2. The customer may not be technically inclined, so keep your language simple and clear. \n3.Avoid making assumptions — ask specific questions to determine the potential causes. You should guide the customer through basic troubleshooting steps and gather data on the situation. \n4. Refine your questions in a strategic way based on the customer's responses for earlier questions. \n5.You should ask questions in an efficient manner, to make the customer satisfied and resolve their problem as quickly as possible. You should also keep your responses short and concise. \n6. If the customer mentions a specific product they are using (for example, ABC electronics), then you are the customer support agent for that product/company, i.e., you represent that product or company and have to take appropriate actions without referring the customer to somewhere else. \n\nYour specific scenario is this: {agent} \n\nPlease start helping the customer now by asking your first question."

class CustomerServiceEnv(BaseEnv):
    name = "customer_service"
    env_config_cls = CustomerServiceEnvConfig

    def __init__(
        self,
        config: CustomerServiceEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: CustomerServiceEnvConfig = config
        dataset_data = self.load_customer_service_dataset(config.dataset_path)
        self.train_scenarios = dataset_data["train"]
        self.eval_scenarios = dataset_data["eval"]
        
        # Inherit prompts from config instance
        self.env_first_message = config.env_first_message
        self.env_default_response = config.env_default_response
        self.judge_prompt_agent = config.judge_prompt_agent
        self.env_optional_message = config.env_optional_message
        self.judge_prompt_suffix = config.judge_prompt_suffix
        self.system_prompt = config.agent_system_prompt
        
        self.episode_outcomes_buffer: List[float] = []
        self.eval_metrics_custom: List[Tuple[str, float]] = []

    @classmethod
    def config_init(cls) -> Tuple[CustomerServiceEnvConfig, List[APIServerConfig]]:
        env_config = CustomerServiceEnvConfig(
            tokenizer_name="Qwen/Qwen2.5-1.5B-Instruct", # TODO: Llama-3.1-8B-Instruct no access
            group_size=1,
            use_wandb=True,
            rollout_server_url="http://localhost:8001",
            wandb_name=cls.name,
            steps_per_eval=50,
            max_episode_turns=20,
            eval_episodes=100,
            temperature = 0.0
        )
        server_configs = [
            # Agent LLM
            APIServerConfig(
                model_name="Qwen/Qwen2.5-1.5B-Instruct",
                base_url="http://localhost:15001/v1",
                api_key="x",
                num_requests_for_eval=128,
                min_p=0.3,
                max_tokens=1024,
                temperature = 0.7
            ),
            # Environment LLM
            APIServerConfig(
                model_name="Qwen/Qwen2.5-1.5B-Instruct",
                base_url="http://localhost:15002/v1",
                api_key="x",
                num_requests_for_eval=128,
                temperature=0.0,  # To be as deterministic as possible
                max_tokens=1024,
            ),
            # Judge LLM
            APIServerConfig(
                model_name="Qwen/Qwen2.5-1.5B-Instruct",
                base_url="http://localhost:15003/v1",
                api_key="x",
                num_requests_for_eval=128,
                temperature=0.0,  # To be as deterministic as possible
                max_tokens=1024,
            ),
        ]
        return env_config, server_configs
    
    def load_customer_service_dataset(self, dataset_path: str) -> Dict:
        """Load customer service scenarios from JSON file"""
        with open(dataset_path, 'r') as f:
            return json.load(f)

    async def collect_trajectory(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataItem], List[Item]]:
        """
        Collects a single trajectory (episode) for the CustomerService environment.
        The 'score' in ScoredDataItem is the final game reward (0.0 to 1.0 based on efficiency).
        """
        seed = item["seed"]
        agent_scenario = item["agent_scenario"]
        env_scenario = item["env_scenario"]
        
        print(f"\n🎯 [EPISODE START] Seed: {seed}")
        print(f"📋 Agent Scenario: {agent_scenario}")
        print(f"🔧 Environment Scenario: {env_scenario}")
        
        agent_messages: List[Message] = []
        env_messages: List[Message] = []
        judge_messages: List[Message] = []
        game_reward = 0.0
        turn = 0
        agent_has_reached_goal = False

        # Set up the system prompts for the agent, environment, and judge
        agent_system_prompt = self.system_prompt.format(agent=agent_scenario)
        agent_messages.append({"role": "system", "content": agent_system_prompt})
        env_system_prompt = self.env_first_message.format(env=env_scenario, agent=agent_scenario)
        env_messages.append({"role": "system", "content": env_system_prompt})
        judge_system_prompt = self.judge_prompt_agent.format(agent=agent_scenario, env=env_scenario)
        judge_messages.append({"role": "system", "content": judge_system_prompt})
        
        print(f"✅ System prompts initialized")

        async with self.server.dedicated_server() as server:
            print(f"🔗 Connected to LLM server")
            
            while turn < self.config.max_episode_turns and not agent_has_reached_goal:
                print(f"\n--- TURN {turn + 1}/{self.config.max_episode_turns} ---")
                
                try:
                    print(f"🤖 Requesting agent response...")
                    chat_completions = await server.chat_completion(
                        messages=agent_messages,
                        n=1,
                    )
                    agent_response = chat_completions.choices[
                        0
                    ].message.content.strip()
                    print(f"🤖 Agent: {agent_response}")
                    logger.info(f"[Seed: {seed}] Agent Response: '{agent_response}'")
                except Exception as e:
                    print(f"❌ Agent LLM API error: {e}")
                    logger.error(f"[Seed: {seed}] Agent LLM API error: {e}")
                    break

                agent_messages.append({"role": "assistant", "content": agent_response})

                message_to_env = agent_response + "\n" + self.env_optional_message.format(env=env_scenario)
                env_messages.append({"role": "user", "content": message_to_env})

                try:
                    print(f"👤 Requesting customer response...")
                    env_response = await server.chat_completion(
                        messages=env_messages,
                        n=1,
                    )
                    env_response = env_response.choices[
                        0
                    ].message.content.strip()
                    print(f"👤 Customer: {env_response}")
                    logger.info(f"[Seed: {seed}] Customer Response: '{env_response}'")
                except Exception as e:
                    print(f"❌ Environment LLM API error: {e}")
                    logger.error(f"[Seed: {seed}] Environment LLM API error: {e}")
                    break
                
                env_messages.append({"role": "assistant", "content": env_response})
                agent_messages.append({"role": "user", "content": env_response})
                
                turn += 1

                # Judge to check if the agent has reached the goal
                env_label = "goal reached" in env_response.lower()
                print(f"🎯 Environment goal check: {'✅ GOAL REACHED' if env_label else '❌ No goal yet'}")
                
                judge_prompt = "The conversation begins here: \n\n"
                
                # Include the full conversation for better context
                for i, msg in enumerate(agent_messages):
                    if msg["role"] == "system":
                        continue  # Skip system message
                    elif msg["role"] == "assistant":
                        judge_prompt += f"Agent: {msg['content']}\n\n"
                    elif msg["role"] == "user":
                        judge_prompt += f"Customer: {msg['content']}\n\n"
                
                judge_prompt += "(End of Conversation)\n\n"
                judge_prompt += self.judge_prompt_suffix.format(env=env_scenario)
                judge_messages.append({"role": "user", "content": judge_prompt})

                # Judge response
                try:
                    print(f"⚖️ Requesting judge evaluation...")
                    judge_response = await server.chat_completion(
                        messages=judge_messages,
                        n=1,
                    )
                    judge_response = judge_response.choices[
                        0
                    ].message.content.strip()
                    print(f"⚖️ Judge: {judge_response}")
                    logger.info(f"[Seed: {seed}] Judge Response: '{judge_response}'")
                except Exception as e:
                    print(f"❌ Judge LLM API error: {e}")
                    logger.error(f"[Seed: {seed}] Judge LLM API error: {e}")
                    break

                judge_messages.append({"role": "assistant", "content": judge_response})
                judge_label = judge_response == '<VALID>'
                print(f"⚖️ Judge evaluation: {'✅ VALID' if judge_label else '❌ NOT VALID'}")
                
                agent_has_reached_goal = judge_label or env_label
                print(f"🏁 Episode complete: {'✅ SUCCESS' if agent_has_reached_goal else '⏳ Continuing...'}")

                if agent_has_reached_goal:
                    break
                
            # Calculate reward based on success/failure and number of turns
            if agent_has_reached_goal:
                # Success: reward decreases with more turns, scaled for max 20 turns
                # 1 turn = 1.0, 10 turns = 0.55, 20 turns = 0.1
                game_reward = max(0.1, 1.0 - (turn - 1) * 0.045)
                print(f"\n🎉 EPISODE SUCCESS!")
                print(f"📊 Turns taken: {turn}")
                print(f"💰 Reward: {game_reward:.3f}")
            else:
                # Failure: no reward
                game_reward = 0.0
                print(f"\n😞 EPISODE FAILED")
                print(f"📊 Max turns reached: {turn}")
                print(f"💰 Reward: {game_reward:.3f}")

        self.episode_outcomes_buffer.append(game_reward) # later get called in eval
        print(f"📈 Added to outcomes buffer (current size: {len(self.episode_outcomes_buffer)})")

        tokenization_result = tokenize_for_trainer(
            tokenizer=self.tokenizer, chat=agent_messages, train_on_all_assistant_turns=True
        )

        tokens = tokenization_result["tokens"]
        masks = tokenization_result["masks"]
        
        print(f"🔢 Tokenization complete: {len(tokens)} tokens, {sum(1 for m in masks if m != -100)} training tokens")
        print(f"🏁 [EPISODE END] Seed: {seed}\n")

        scored_data_item = ScoredDataItem(
            messages=agent_messages if self.config.include_messages else None,
            tokens=tokens,
            masks=masks,
            scores=game_reward,
        )
        return scored_data_item, []

    async def get_next_item(self) -> Item:
        scenario = random.choice(self.train_scenarios)
        return {
            "seed": random.randint(0, 1_000_000),
            "agent_scenario": scenario["agent"],
            "env_scenario": scenario["env"]
        }

    async def get_eval_item(self) -> Item:
        scenario = random.choice(self.eval_scenarios)
        return {
            "seed": random.randint(1_000_001, 2_000_000),
            "agent_scenario": scenario["agent"],
            "env_scenario": scenario["env"]
        }

    async def setup(self):
        logger.info(f"Setting up {self.name} environment.")

    async def evaluate(self, *args, **kwargs):
        logger.info(
            f"Starting evaluation for {self.name} with {self.config.eval_episodes} episodes."
        )

        wins = 0
        losses = 0

        eval_outcomes: List[float] = []

        for i in range(self.config.eval_episodes):
            item = await self.get_eval_item()
            scored_item_tuple = await self.collect_trajectory(item)
            if scored_item_tuple and scored_item_tuple[0]:
                outcome = scored_item_tuple[0]["scores"]
                eval_outcomes.append(outcome)
            else:
                logger.warning(
                    f"Evaluation episode {i+1} (seed {item['seed']}) failed to produce data."
                )

        if not eval_outcomes:
            logger.warning("No evaluation episodes completed successfully.")
            self.eval_metrics_custom = []
            return

        for outcome in eval_outcomes:
            if outcome > 0:  # Any positive reward = success
                wins += 1
            else:  # Zero reward = failure
                losses += 1

        num_completed = len(eval_outcomes)
        win_rate = wins / num_completed if num_completed > 0 else 0
        loss_rate = losses / num_completed if num_completed > 0 else 0
        avg_reward = sum(eval_outcomes) / num_completed if num_completed > 0 else 0

        self.eval_metrics_custom = [
            (f"{self.name}_eval/win_rate", win_rate),  # Success percentage
            (f"{self.name}_eval/loss_rate", loss_rate),  # Failure percentage
            (f"{self.name}_eval/avg_reward", avg_reward),  # Average quality
            (f"{self.name}_eval/num_completed_episodes", num_completed),
        ]
        logger.info(
            f"Evaluation completed for {self.name}. Metrics: {self.eval_metrics_custom}"
        )

    async def wandb_log(self, wandb_metrics: Optional[Dict[str, float]] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        if self.episode_outcomes_buffer:
            avg_training_reward = sum(self.episode_outcomes_buffer) / len(
                self.episode_outcomes_buffer
            )
            wandb_metrics[f"{self.name}_train/avg_episode_reward"] = avg_training_reward
            train_wins = sum(1 for r in self.episode_outcomes_buffer if r > 0)
            train_losses = sum(1 for r in self.episode_outcomes_buffer if r == 0)
            wandb_metrics[f"{self.name}_train/win_count"] = train_wins
            wandb_metrics[f"{self.name}_train/loss_count"] = train_losses
            wandb_metrics[f"{self.name}_train/num_episodes_in_batch"] = len(
                self.episode_outcomes_buffer
            )

        self.episode_outcomes_buffer = []

        for key, value in self.eval_metrics_custom:
            wandb_metrics[key] = value
        self.eval_metrics_custom = []

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    CustomerServiceEnv.cli()