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

# TODO: change accordingly to twenty questions
# Set up file logging for console output
def setup_file_logging():
    """Set up file logging for the twenty questions environment"""
    # Create logs directory in the same folder as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"twenty_questions_rollouts_{timestamp}.log")
    
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
    rollout_logger = logging.getLogger('twenty_questions_rollouts')
    rollout_logger.setLevel(logging.INFO)
    rollout_logger.addHandler(file_handler)
    rollout_logger.addHandler(console_handler)
    
    return rollout_logger, log_file

class TwentyQuestionsEnvConfig(BaseEnvConfig):
    """
    Configuration for the Twenty Questions environment.
    """

    env_name: str = "TwentyQuestions"
    max_episode_turns: int = 20
    eval_episodes: int = 100 # TODO: experiment to adjust
    dataset_path: str = "/shared/share_mala/melody/atropos/environments/game_environments/twenty_questions/twenty_questions_dataset.json"
    env_first_message: str = "You are the environment for a game of 20 Questions. You will be given a topic (e.g., cat, deer, Abraham Lincoln), and your role is to answer 'Yes' or 'No' to questions about the topic. Respond strictly with 'Yes' or 'No' only, based on the truth of the matter. Do not provide any additional information. If the user guesses the correct answer, respond with 'Goal reached.' Enforce a strict match before saying 'Goal reached.', for example, if the user guesses 'Is the occupation Engineer?' and the topic was software engineering, say 'No'. However, if the user guesses 'Is the occupation a type of engineering?', say 'Yes'. Wait for the user's questions and respond accordingly.\n\nFor example:\n- User: 'Is this person alive?'\n- You: 'No'\n- User: 'Is this a living creature?'\n- You: 'Yes'\n\nThe player is trying to guess a {agent}, and the correct answer is {env}."
    env_default_response: str = "Sorry, I cannot answer this question. You should only ask questions that can be answered with yes or no. Please ask a different question."
    judge_prompt_agent: str = "You are an expert judge for the game of 20 questions. You will be given a question asked by the player, and you need to judge if it successfully guesses the correct topic.\n In particular: Check that the player has precisely guessed {env} in their question. Reply with <VALID> or <NOTVALID>. Reply with <VALID> if and only if the given question successfully ends the game by guessing the correct {agent}, which is: {env}, otherwise reply with <NOTVALID>.\n\nExample 1: Topic to Guess: Car\nPlayer: Is the invention a car?\n Answer: <VALID>\n\nExample 2: Topic to Guess: Car\nPlayer: Does the invention have wheels?\nAnswer: <NOTVALID>\n\n Example 3: Topic to Guess: Apple, Inc.\n Player: Does this company produce IPhones? Answer: <NOTVALID>"
    env_optional_message: str = "(Remember the player is trying to guess a {agent}, and the correct answer is {env}. Reply with Yes or No in relation to {env}.)"
    judge_prompt_suffix: str = "\n\nNow judge whether the player has successfully guessed the correct {agent}, which is {env} in this particular game. Reply with <VALID> only if the player has guessed {env} in the question, otherwise reply with <NOTVALID>. Note that guessing a particular characteristics of {env} is not enough, the player needs to arrive at the final answer in order for you to reply with <VALID>.\n\nAnswer: "
    agent_system_prompt: str = "You are playing a game of 20 Questions. Your goal is to guess the name of a thing or person by asking up to 20 yes-or-no questions. After each question, you will receive an answer: 'Yes' or 'No.' Use the answers provided to refine your guesses.\n\nHere are your instructions:\n- You can ask only yes-or-no questions.\n- After receiving each answer, you should adapt your questions based on the new information.\n- Your goal is to guess the topic in as few questions as possible.\n- If you're confident, you can make a guess before reaching 20 questions.\n\nThe game starts now. You are trying to guess a {agent}. Ask your first question!"
    gpu_memory_utilization: float = 0.2

class TwentyQuestionsEnv(BaseEnv):
    name = "twenty_questions"
    env_config_cls = TwentyQuestionsEnvConfig

    def __init__(
        self,
        config: TwentyQuestionsEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: TwentyQuestionsEnvConfig = config
        dataset_data = self.load_twenty_questions_dataset(config.dataset_path)
        self.train_scenarios = dataset_data["train"]
        self.eval_scenarios = dataset_data["eval"]
        
        # Set up file logging
        self.rollout_logger, self.log_file = setup_file_logging()
        self.rollout_logger.info(f"🚀 Twenty Questions Environment initialized")
        self.rollout_logger.info(f"📝 Rollouts will be logged to: {self.log_file}")
        
        # Inherit prompts from config instance
        self.env_first_message = config.env_first_message
        self.env_default_response = config.env_default_response
        self.judge_prompt_agent = config.judge_prompt_agent
        self.env_optional_message = config.env_optional_message
        self.judge_prompt_suffix = config.judge_prompt_suffix
        self.system_prompt = config.agent_system_prompt
        
        self.episode_outcomes_buffer: List[float] = []
        self.eval_metrics_custom: List[Tuple[str, float]] = []

    # TODO: check twenty questions temperature, max tokens, top p, min p, etc. settings
    @classmethod
    def config_init(cls) -> Tuple[TwentyQuestionsEnvConfig, List[APIServerConfig]]:
        env_config = TwentyQuestionsEnvConfig(
            tokenizer_name="Qwen-2.5-7B-Instruct", # TODO: Llama-3.1-8B-Instruct no access
            group_size=1,
            use_wandb=True,
            rollout_server_url="http://localhost:8001",
            wandb_name=cls.name,
            steps_per_eval=50,
            max_episode_turns=20,
            eval_episodes=100,
            temperature = 0.0,
        )
        server_configs = [
            # Agent LLM
            APIServerConfig(
                model_name="Qwen-2.5-7B-Instruct",
                base_url="http://localhost:15001/v1",
                api_key="x",
                num_requests_for_eval=128,
                min_p=0.3,
                max_tokens=1024,
                temperature=1.5  # Higher temperature for diverse training data
            ),
            # Environment LLM
            APIServerConfig(
                model_name="gpt-4o-mini",
                base_url="https://api.openai.com/v1",
                api_key=os.getenv("OPENAI_API_KEY", "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"),
                num_requests_for_eval=128,
                temperature=0.0,  # To be as deterministic as possible
                max_tokens=1024,
            ),
            # Judge LLM
            APIServerConfig(
                model_name="gpt-4o-mini",
                base_url="https://api.openai.com/v1",
                api_key=os.getenv("OPENAI_API_KEY", "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"),
                num_requests_for_eval=128,
                temperature=0.0,  # To be as deterministic as possible
                max_tokens=1024,
            ),
        ]
        return env_config, server_configs
    
    def load_twenty_questions_dataset(self, dataset_path: str) -> Dict:
        """Load twenty questions scenarios from JSON file"""
        with open(dataset_path, 'r') as f:
            return json.load(f)

    def environment_response_extractor(self, env_response: str) -> str:
        """Extract the environment response from the LLM response"""
        response = env_response.lower()
        if response == "yes" or response == "yes.":
            return "Yes"
        elif response == "no" or response == "no.":
            return "No"
        elif response == "goal reached" or response == "goal reached.":
            return "Goal reached"
        else:
            raise ValueError(f"Given response {response} not valid.")

        return env_response

    async def collect_trajectory(
        self, item: Item, is_eval: bool = False
    ) -> Tuple[Optional[ScoredDataItem], List[Item]]:
        """
        Collects a single trajectory (episode) for the TwentyQuestions environment.
        The 'score' in ScoredDataItem is the final game reward (0.0 to 1.0 based on efficiency).
        """
        seed = item["seed"]
        agent_type = item["agent_type"]
        env_word = item["env_word"]
        
        self.rollout_logger.info(f"\n🎯 [EPISODE START] Seed: {seed}")
        self.rollout_logger.info(f"📋 Agent Type: {agent_type}")
        self.rollout_logger.info(f"🔧 Environment Word: {env_word}")
        self.rollout_logger.info(f"🌡️ Mode: {'EVALUATION (temp=0.7)' if is_eval else 'TRAINING (temp=1.5)'}")
        
        agent_messages: List[Message] = []
        env_messages: List[Message] = []
        judge_messages: List[Message] = []
        game_reward = 0.0
        turn = 0
        agent_has_reached_goal = False

        # Set up the system prompts for the agent, environment, and judge
        agent_system_prompt = self.system_prompt.format(agent=agent_type)
        agent_messages.append({"role": "system", "content": agent_system_prompt})
        env_system_prompt = self.env_first_message.format(env=env_word, agent=agent_type)
        env_messages.append({"role": "system", "content": env_system_prompt})
        judge_system_prompt = self.judge_prompt_agent.format(agent=agent_type, env=env_word)
        judge_messages.append({"role": "system", "content": judge_system_prompt})
        
        self.rollout_logger.info(f"✅ System prompts initialized")

        async with self.server.dedicated_server() as server:
            self.rollout_logger.info(f"🔗 Connected to LLM server")
            
            while turn < self.config.max_episode_turns and not agent_has_reached_goal:
                self.rollout_logger.info(f"\n--- TURN {turn + 1}/{self.config.max_episode_turns} ---")
                
                try:
                    self.rollout_logger.info(f"🤖 Requesting agent response...")
                    chat_completions = await server.chat_completion(
                        messages=agent_messages,
                        n=1,
                        temperature=0.7 if is_eval else 1.5,  # Dynamic temperature based on eval mode
                    )
                    agent_response = chat_completions.choices[
                        0
                    ].message.content.strip()
                    self.rollout_logger.info(f"[Seed: {seed}] Agent Response: '{agent_response}'")
                except Exception as e:
                    self.rollout_logger.error(f"[Seed: {seed}] Agent LLM API error: {e}")
                    break

                agent_messages.append({"role": "assistant", "content": agent_response})

                message_to_env = agent_response + "\n" + self.env_optional_message.format(env=env_word)
                env_messages.append({"role": "user", "content": message_to_env})

                try:
                    self.rollout_logger.info(f"👤 Requesting environment response...")
                    env_response_raw = await server.chat_completion(
                        messages=env_messages,
                        n=1,
                    )
                    env_response_raw = env_response_raw.choices[
                        0
                    ].message.content.strip()
                    self.rollout_logger.info(f"[Seed: {seed}] Raw Environment Response: '{env_response_raw}'")
                    
                    # API call successful, extract valid response
                    env_response = self.environment_response_extractor(env_response_raw)
                    self.rollout_logger.info(f"[Seed: {seed}] ✅ Environment Response: '{env_response}'")
                    
                except Exception as e:
                    # API call failed, use default response
                    env_response = self.env_default_response
                    self.rollout_logger.error(f"[Seed: {seed}] Environment LLM API error, using default response: {e}")

                env_messages.append({"role": "assistant", "content": env_response})
                agent_messages.append({"role": "user", "content": env_response})
                
                turn += 1

                # Judge to check if the agent has reached the goal
                env_label = "goal reached" in env_response.lower()
                self.rollout_logger.info(f"🎯 Environment goal check: {'✅ GOAL REACHED' if env_label else '❌ No goal yet'}")
                
                judge_prompt = "The conversation begins here: \n\n"
                
                # Include agent messages for judging (skip system and user messages)
                for i, msg in enumerate(agent_messages):
                    if msg["role"] == "system":
                        continue  # Skip system message
                    elif msg["role"] == "assistant":
                        judge_prompt += f"Agent: {msg['content']}\n\n"
                    elif msg["role"] == "user":
                        continue  # Skip environment responses for judging
                
                judge_prompt += "(End of Agent Turn)\n\n"
                judge_prompt += self.judge_prompt_suffix.format(env=env_word)
                judge_messages.append({"role": "user", "content": judge_prompt})

                # Judge response
                try:
                    self.rollout_logger.info(f"⚖️ Requesting judge evaluation...")
                    judge_response = await server.chat_completion(
                        messages=judge_messages,
                        n=1,
                    )
                    judge_response = judge_response.choices[
                        0
                    ].message.content.strip()
                    self.rollout_logger.info(f"[Seed: {seed}] Judge Response: '{judge_response}'")
                except Exception as e:
                    self.rollout_logger.error(f"[Seed: {seed}] Judge LLM API error: {e}")
                    break

                judge_messages.append({"role": "assistant", "content": judge_response})
                judge_label = judge_response == '<VALID>'
                self.rollout_logger.info(f"⚖️ Judge evaluation: {'✅ VALID' if judge_label else '❌ NOT VALID'}")
                
                agent_has_reached_goal = judge_label or env_label
                self.rollout_logger.info(f"🏁 Episode complete: {'✅ SUCCESS' if agent_has_reached_goal else '⏳ Continuing...'}")

                if agent_has_reached_goal:
                    break
                
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
        self.rollout_logger.info(f"🏁 [EPISODE END] Seed: {seed}\n")

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
            "agent_type": scenario["agent"],
            "env_word": scenario["env"]
        }

    async def get_eval_item(self) -> Item:
        scenario = random.choice(self.eval_scenarios)
        return {
            "seed": random.randint(1_000_001, 2_000_000),
            "agent_type": scenario["agent"],
            "env_word": scenario["env"]
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