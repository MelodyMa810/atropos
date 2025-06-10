import json
import logging
import random
from typing import Dict, List, Optional, Tuple
import os
from datetime import datetime

from atroposlib.envs.base import APIServerConfig, BaseEnv, BaseEnvConfig, ScoredDataItem
from atroposlib.type_definitions import Item, Message
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer
from atroposlib.utils.tool_call_parser import parse_tool_call

logger = logging.getLogger(__name__)

# Set up file logging for console output
def setup_file_logging():
    """Set up file logging for the customer service environment"""
    # Create logs directory in the same folder as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"customer_service_rollouts_{timestamp}.log")
    
    # Create a file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Create a logger for rollouts
    rollout_logger = logging.getLogger('customer_service_rollouts')
    rollout_logger.setLevel(logging.INFO)
    rollout_logger.addHandler(file_handler)
    rollout_logger.addHandler(console_handler)
    
    return rollout_logger, log_file

class CustomerServiceEnvConfig(BaseEnvConfig):
    """
    Configuration for the CustomerService environment.
    """

    env_name: str = "CustomerService"
    max_episode_turns: int = 20
    eval_episodes: int = 100 # TODO: experiment to adjust
    dataset_path: str = "/shared/share_mala/melody/atropos/environments/game_environments/customer_service/customer_service_dataset.json"
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
        
        # Set up file logging
        self.rollout_logger, self.log_file = setup_file_logging()
        self.rollout_logger.info(f"🚀 Customer Service Environment initialized")
        self.rollout_logger.info(f"📝 Rollouts will be logged to: {self.log_file}")
        
        # Rollout tracking
        self.rollout_counter = 0
        self.batch_counter = 0
        self.total_rollouts_completed = 0
        
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
            tokenizer_name="Qwen/Qwen2.5-1.5B-Instruct", # Fixed to match model name
            group_size=1,  # This controls how many rollouts happen in parallel
            use_wandb=True,
            rollout_server_url="http://localhost:8001",
            wandb_name=cls.name,
            steps_per_eval=50,  # Changed from 50 to 1 for testing
            max_episode_turns=20,
            eval_episodes=100,  # Changed from 100 to 1 for testing
            temperature = 0.0,
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
                temperature = 1.5
                # TODO: double check temperature settings
            ),
            # Environment LLM
            APIServerConfig(
                model_name="gpt-4o-mini",x
                base_url="https://api.openai.com/v1",
                api_key=os.getenv("OPENAI_API_KEY"),
                # model_name="qwen/qwen-2.5-7b-instruct:free",  # Free model on OpenRouter
                # base_url="https://openrouter.ai/api/v1",
                # api_key=os.getenv("OPENROUTER_API_KEY"),  # Removed hardcoded key
                num_requests_for_eval=128,
                temperature=0.0,  # To be as deterministic as possible
                max_tokens=1024,
            ),
            # Judge LLM
            APIServerConfig(
                model_name="gpt-4o-mini",
                base_url="https://api.openai.com/v1",
                api_key=os.getenv("OPENAI_API_KEY"),
                # model_name="deepseek/deepseek-r1:free",  # Free model on OpenRouter
                # base_url="https://openrouter.ai/api/v1",
                # api_key=os.getenv("OPENROUTER_API_KEY"),  # Removed hardcoded key
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
        self, item: Item, is_eval: bool = False
    ) -> Tuple[Optional[ScoredDataItem], List[Item]]:
        """
        Collects a single trajectory (episode) for the CustomerService environment.
        The 'score' in ScoredDataItem is the final game reward (0.0 to 1.0 based on efficiency).
        """
        if not is_eval:
            self.rollout_counter += 1
            self.total_rollouts_completed += 1
        
        seed = item["seed"]
        agent_scenario = item["agent_scenario"]
        env_scenario = item["env_scenario"]
        
        mode_str = "EVALUATION" if is_eval else f"TRAINING ROLLOUT #{self.rollout_counter}"
        self.rollout_logger.info(f"\n🎯 [EPISODE START] {mode_str} | Seed: {seed}")
        self.rollout_logger.info(f"📊 Total rollouts completed: {self.total_rollouts_completed}")
        self.rollout_logger.info(f"📋 Agent Scenario: {agent_scenario}")
        self.rollout_logger.info(f"🔧 Environment Scenario: {env_scenario}")
        self.rollout_logger.info(f"🌡️ Mode: {'EVALUATION (temp=0.7)' if is_eval else 'TRAINING (temp=1.5)'}")
        
        agent_messages: List[Message] = []
        env_messages: List[Message] = []
        game_reward = 0.0
        turn = 0
        agent_has_reached_goal = False

        # Set up the system prompts for the agent and environment
        agent_system_prompt = self.system_prompt.format(agent=agent_scenario)
        agent_messages.append({"role": "system", "content": agent_system_prompt})
        env_system_prompt = self.env_first_message.format(env=env_scenario, agent=agent_scenario)
        env_messages.append({"role": "system", "content": env_system_prompt})
        
        self.rollout_logger.info(f"✅ System prompts initialized")

        # Get dedicated server connections for each role
        # Server index 0: Agent LLM (Qwen), Server index 1: Environment LLM (OpenAI)
        agent_server = self.server.servers[0]
        env_server = self.server.servers[1]  
        
        self.rollout_logger.info(f"🔗 Connected to LLM servers")
        self.rollout_logger.info(f"🤖 Agent server: {agent_server.config.model_name} at {agent_server.config.base_url}")
        self.rollout_logger.info(f"👤 Environment server: {env_server.config.model_name} at {env_server.config.base_url}")
        
        while turn < self.config.max_episode_turns and not agent_has_reached_goal:
            self.rollout_logger.info(f"\n--- TURN {turn + 1}/{self.config.max_episode_turns} ---")
            
            try:
                self.rollout_logger.info(f"🤖 Requesting agent response...")
                async with agent_server.sem:
                    chat_completions = await agent_server.chat_completion(
                        messages=agent_messages,
                        n=1,
                        temperature=0.7 if is_eval else 1.5,  # Dynamic temperature based on eval mode
                    )
                agent_response = chat_completions.choices[0].message.content.strip()
                self.rollout_logger.info(f"[Seed: {seed}] Agent Response: '{agent_response}'")
            except Exception as e:
                self.rollout_logger.error(f"[Seed: {seed}] Agent LLM API error: {e}")
                break

            agent_messages.append({"role": "assistant", "content": agent_response})

            message_to_env = agent_response + "\n" + self.env_optional_message.format(env=env_scenario)
            env_messages.append({"role": "user", "content": message_to_env})

            try:
                self.rollout_logger.info(f"👤 Requesting customer response...")
                async with env_server.sem:
                    env_response = await env_server.chat_completion(
                        messages=env_messages,
                        n=1,
                    )
                env_response = env_response.choices[0].message.content.strip()
                self.rollout_logger.info(f"[Seed: {seed}] Customer Response: '{env_response}'")
            except Exception as e:
                self.rollout_logger.error(f"[Seed: {seed}] Environment LLM API error: {e}")
                break
            
            env_messages.append({"role": "assistant", "content": env_response})
            agent_messages.append({"role": "user", "content": env_response})
            
            turn += 1

            # Check if the agent has reached the goal
            agent_has_reached_goal = await self.check_agent_reached_goal(env_response, agent_messages, env_scenario, agent_scenario)
            self.rollout_logger.info(f"🏁 Agent has reached goal: {agent_has_reached_goal}")

            # Extra verification if the environment thinks the game is solved
            if agent_has_reached_goal:
                judge_label = await self.run_judge_verification(agent_messages, env_scenario, agent_scenario)
                if not judge_label:
                    agent_has_reached_goal = False
                    self.rollout_logger.info(f"🏁 Episode terminated: ❌ FAILURE (Environment thought goal reached but Judge disagreed)")
                    break
                else:
                    self.rollout_logger.info(f"🏁 Episode complete: ✅ SUCCESS (Judge confirmed)")
                    break
            else:
                self.rollout_logger.info(f"🏁 Turn complete: ⏳ Continuing...")

        # Calculate reward based on success/failure and number of turns
        if agent_has_reached_goal:
            # Success: reward decreases with more turns, scaled for max 20 turns
            # 1 turn = 1.0, 10 turns = 0.55, 20 turns = 0.1
            game_reward = max(0.1, 1.0 - (turn - 1) * 0.045)
            self.rollout_logger.info(f"\n🎉 EPISODE SUCCESS!")
            self.rollout_logger.info(f"📊 Turns taken: {turn}")
            self.rollout_logger.info(f"💰 Reward: {game_reward:.3f}")
        else:
            # Failure: no reward
            game_reward = 0.0
            self.rollout_logger.info(f"\n😞 EPISODE FAILED")
            self.rollout_logger.info(f"📊 Max turns reached: {turn}")
            self.rollout_logger.info(f"💰 Reward: {game_reward:.3f}")

        self.episode_outcomes_buffer.append(game_reward) # later get called in eval
        self.rollout_logger.info(f"📈 Added to outcomes buffer (current size: {len(self.episode_outcomes_buffer)})")

        tokenization_result = tokenize_for_trainer(
            tokenizer=self.tokenizer, chat=agent_messages, train_on_all_assistant_turns=True
        )

        tokens = tokenization_result["tokens"]
        masks = tokenization_result["masks"]
        
        self.rollout_logger.info(f"🔢 Tokenization complete: {len(tokens)} tokens, {sum(1 for m in masks if m != -100)} training tokens")
        self.rollout_logger.info(f"🏁 [EPISODE END] {mode_str} | Seed: {seed}\n")

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
            scored_item_tuple = await self.collect_trajectory(item, is_eval=True)
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
            self.batch_counter += 1
            batch_size = len(self.episode_outcomes_buffer)
            
            # Log batch completion
            self.rollout_logger.info(f"\n📦 [BATCH #{self.batch_counter} COMPLETE]")
            self.rollout_logger.info(f"📊 Rollouts in this batch: {batch_size}")
            self.rollout_logger.info(f"📈 Total training rollouts completed: {self.total_rollouts_completed}")
            
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
            
            # Log batch metrics
            self.rollout_logger.info(f"🏆 Batch metrics - Wins: {train_wins}, Losses: {train_losses}, Avg Reward: {avg_training_reward:.3f}")

        self.episode_outcomes_buffer = []

        for key, value in self.eval_metrics_custom:
            wandb_metrics[key] = value
        self.eval_metrics_custom = []

        await super().wandb_log(wandb_metrics)

    async def check_agent_reached_goal(self, env_response: str, agent_messages: List[Message], env_scenario: str, agent_scenario: str) -> bool:
        """
        Check if the agent has reached the goal based on environment response and conversation context.
        
        Args:
            env_response: The latest response from the environment/customer
            agent_messages: The conversation history with the agent
            env_scenario: The solution scenario for reference
            agent_scenario: The agent scenario for context
            
        Returns:
            bool: True if goal is reached, False otherwise
        """
        # Primary check: Environment explicitly says goal is reached
        env_label = "goal reached" in env_response.lower()
        self.rollout_logger.info(f"🎯 Environment goal check: env_label = {env_label}")
        verification_label = await self.run_judge_verification(agent_messages, env_scenario, agent_scenario)
        self.rollout_logger.info(f"⚖️ Judge verification check: verification_label = {verification_label}")
        
        final_result = env_label or verification_label
        if final_result:
            if env_label and verification_label:
                reason = "Both environment and judge confirmed goal reached"
            elif env_label:
                reason = "Environment confirmed goal reached (judge verification also ran)"
            elif verification_label:
                reason = "Judge confirmed goal reached (environment did not detect 'goal reached')"
        else:
            reason = "Neither environment nor judge confirmed goal reached"
            
        self.rollout_logger.info(f"📊 Final goal check result: {final_result} ({reason})")
        return final_result
        

    async def run_judge_verification(self, agent_messages: List[Message], env_scenario: str, agent_scenario: str) -> bool:
        """
        Run the judge verification process.
        
        Args:
            agent_messages: The conversation history with the agent
            env_scenario: The solution scenario for reference
            agent_scenario: The agent scenario for context
            
        Returns:
            bool: True if judge validates the solution, False otherwise
        """
        judge_server = self.server.servers[2]
        self.rollout_logger.info(f"🔗 Connected to judge server: {judge_server.config.base_url}")
        judge_messages: List[Message] = []
        judge_system_prompt = self.config.judge_prompt_agent.format(agent=agent_scenario, env=env_scenario)
        judge_messages.append({"role": "system", "content": judge_system_prompt})
        
        judge_prompt = "The conversation begins here: \n\n"
        
        # Include the full conversation for better context
        for i, msg in enumerate(agent_messages):
            if msg["role"] == "system":
                continue  # Skip system message
            elif msg["role"] == "assistant":
                judge_prompt += f"Agent: {msg['content']}\n\n"
            elif msg["role"] == "user":
                continue # Skip environment responses
        
        judge_prompt += "(End of Agent Turn)\n\n"
        judge_prompt += self.config.judge_prompt_suffix.format(env=env_scenario)
        judge_messages.append({"role": "user", "content": judge_prompt})

        # Judge response
        try:
            self.rollout_logger.info(f"⚖️ Requesting judge evaluation...")
            async with judge_server.sem:
                judge_response = await judge_server.chat_completion(
                    messages=judge_messages,
                    n=1,
                )
            judge_response = judge_response.choices[0].message.content.strip()
            self.rollout_logger.info(f"Judge Response: '{judge_response}'")
        except Exception as e:
            self.rollout_logger.error(f"Judge LLM API error: {e}")
            self.rollout_logger.info(f"📊 Judge verification result: FALSE (API error)")
            return False

        judge_messages.append({"role": "assistant", "content": judge_response})
        judge_label = judge_response == '<VALID>'
        self.rollout_logger.info(f"📊 Judge verification result: {judge_label} (Response was '{judge_response}')")
        
        return judge_label

if __name__ == "__main__":
    CustomerServiceEnv.cli()