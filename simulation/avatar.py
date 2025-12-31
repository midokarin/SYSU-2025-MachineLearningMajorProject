from simulation.base.abstract_avatar import abstract_avatar
from simulation.memory import AvatarMemory

from termcolor import colored, cprint
import openai
import os

import re
import numpy as np
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain.chat_models import ChatOpenAI
from simulation.retriever import AvatarRetriver
import time
import datetime
import torch
from langchain.embeddings import OpenAIEmbeddings
import pandas as pd

import wandb

import simulation.vars as vars


def _resolve_chat_model(args=None):
    """Resolve chat model name from args/env, avoiding hard-coded GPT models."""
    return (
        os.getenv("DEEPSEEK_CHAT_MODEL")
        or os.getenv("LLM_CHAT_MODEL")
        or os.getenv("OPENAI_MODEL")
        or (getattr(args, "llm_chat_model", None) if args is not None else None)
        or (getattr(args, "chat_model", None) if args is not None else None)
        or "gpt-3.5-turbo"  # fallback only
    )


class Avatar(abstract_avatar):
    def __init__(self, args, avatar_id, init_property, init_statistic):
        super().__init__(args, avatar_id)

        self.parse_init_property(init_property)
        self.parse_init_statistic(init_statistic)
        self.log_file = f"storage/{args.dataset}/{args.modeltype}/{args.simulation_name}/running_logs/{avatar_id}.txt"
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        self.init_memory()

    def parse_init_property(self, init_property):
        self.taste = init_property["taste"].split("| ")
        self.high_rating = init_property["high_rating"]

    def parse_init_statistic(self, init_statistic):
        """
        Parse the init statistic of the avatar
        """
        activity_dict = {
            1: "An Incredibly Elusive Occasional Viewer, so seldom attracted by movie recommendations that it's almost a legendary event when you do watch a movie. Your movie-watching habits are extraordinarily infrequent. And you will exit the recommender system immediately even if you just feel little unsatisfied.",
            2: "An Occasional Viewer, seldom attracted by movie recommendations. Only curious about watching movies that strictly align the taste. The movie-watching habits are not very infrequent. And you tend to exit the recommender system if you have a few unsatisfied memories.",
            3: "A Movie Enthusiast with an insatiable appetite for films, willing to watch nearly every movie recommended to you. Movies are a central part of your life, and movie recommendations are integral to your existence. You are tolerant of recommender system, which means you are not easy to exit recommender system even if you have some unsatisfied memory.",
        }
        conformity_dict = {
            1: "A Dedicated Follower who gives ratings heavily relies on movie historical ratings, rarely expressing independent opinions. Usually give ratings that are same as historical ratings. ",
            2: "A Balanced Evaluator who considers both historical ratings and personal preferences when giving ratings to movies. Sometimes give ratings that are different from historical rating.",
            3: "A Maverick Critic who completely ignores historical ratings and evaluates movies solely based on own taste. Usually give ratings that are a lot different from historical ratings.",
        }
        diversity_dict = {
            1: "An Exceedingly Discerning Selective Viewer who watches movies with a level of selectivity that borders on exclusivity. The movie choices are meticulously curated to match personal taste, leaving no room for even a hint of variety.",
            2: "A Niche Explorer who occasionally explores different genres and mostly sticks to preferred movie types.",
            3: "A Cinematic Trailblazer, a relentless seeker of the unique and the obscure in the world of movies. The movie choices are so diverse and avant-garde that they defy categorization.",
        }

        self.conformity_group = init_statistic["conformity"]
        self.activity_group = init_statistic["activity"]
        self.diversity_group = init_statistic["diversity"]
        self.conformity_dsc = conformity_dict[self.conformity_group]
        self.activity_dsc = activity_dict[self.activity_group]
        self.diversity_dsc = diversity_dict[self.diversity_group]

    # =========================
    # Ablation helpers (NEW)
    # =========================
    def _ablate_mode(self) -> str:
        """
        profile ablation mode from args.
        Supported: none, no_profile, no_social, no_taste,
                   no_activity, no_conformity, no_diversity
        """
        return getattr(self.args, "ablate_profile", "none")

    def _include_activity(self) -> bool:
        mode = self._ablate_mode()
        return mode not in ("no_profile", "no_social", "no_activity")

    def _include_conformity(self) -> bool:
        mode = self._ablate_mode()
        return mode not in ("no_profile", "no_social", "no_conformity")

    def _include_diversity(self) -> bool:
        mode = self._ablate_mode()
        return mode not in ("no_profile", "no_social", "no_diversity")

    def _include_taste(self) -> bool:
        mode = self._ablate_mode()
        return mode not in ("no_profile", "no_taste")

    def _include_rating_tendency(self) -> bool:
        mode = self._ablate_mode()
        return mode not in ("no_profile", "no_taste")

    def _build_profile_text(
        self,
        include_activity=True,
        include_conformity=True,
        include_diversity=True,
        include_taste=True,
        include_rating_tendency=True,
    ) -> str:
        """
        Build profile text according to ablation mode.
        IMPORTANT: This only controls what is shown to LLM in the prompt,
                   not the underlying sampled groups.
        """
        mode = self._ablate_mode()

        # Highest priority: remove entire profile
        if mode == "no_profile":
            return ""

        # Global removals
        if mode == "no_social":
            include_activity = include_conformity = include_diversity = False
        if mode == "no_taste":
            include_taste = include_rating_tendency = False

        # Single-trait removals
        if mode == "no_activity":
            include_activity = False
        if mode == "no_conformity":
            include_conformity = False
        if mode == "no_diversity":
            include_diversity = False

        parts = []

        # Traits
        if include_activity:
            parts.append(f"Your activity trait is described as: {self.activity_dsc}")
        if include_conformity:
            parts.append(f"Your conformity trait is described as: {self.conformity_dsc}")
        if include_diversity:
            parts.append(f"Your diversity trait is described as: {self.diversity_dsc}")

        # Tastes
        if include_taste:
            parts.append(f"Beyond that, your movie tastes are: {'; '.join(self.taste).replace('I ','')}. ")

        # Rating tendency
        if include_rating_tendency:
            try:
                high_rating = self.high_rating.replace("You are", "")
            except Exception:
                high_rating = ""
            parts.append(f"And your rating tendency is {high_rating}")

        # Only explain characteristics that actually exist in prompt
        if include_activity or include_conformity or include_diversity:
            parts.append(
                "The activity characteristic pertains to the frequency of your movie-watching habits. "
                "The conformity characteristic measures the degree to which your ratings are influenced by historical ratings. "
                "The diversity characteristic gauges your likelihood of watching movies that may not align with your usual taste."
            )

        if not parts:
            return ""
        return "\n" + "\n".join(parts)

    def init_memory(self):
        """
        Initialize the memory of the avatar
        """
        t1 = time.time()

        def score_normalizer(val: float) -> float:
            return 1 - 1 / (1 + np.exp(val))

        # NOTE: embeddings 用 HF（离线更稳），如需改为 DeepSeek embeddings 可再适配
        embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        embedding_size = 384  # all-MiniLM-L6-v2 的向量维度是 384
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(
            embeddings_model.embed_query,
            index,
            InMemoryDocstore({}),
            {},
            relevance_score_fn=score_normalizer,
        )

        # NOTE: ChatOpenAI 走 OpenAI-compatible（DeepSeek 也可以）
        LLM = ChatOpenAI(
            max_tokens=1000,
            temperature=0.3,
            request_timeout=30,
            model_name=_resolve_chat_model(self.args),
        )

        avatar_retriever = AvatarRetriver(vectorstore=vectorstore, k=5)
        self.memory = AvatarMemory(
            memory_retriever=avatar_retriever,
            llm=LLM,
            reflection_threshold=3,
            use_wandb=self.use_wandb,
        )

        t2 = time.time()
        cprint(f"Avatar {self.avatar_id} is initialized with memory", color="green", attrs=["bold"])
        cprint(f"Time cost: {t2 - t1}s", color="green", attrs=["bold"])

    def _reaction(self, messages=None, timeout=30):
        """
        Summarize the feelings of the avatar for recommended item list.
        """
        response = ""
        except_waiting_time = 1
        max_waiting_time = 16
        current_sleep_time = 0.5

        while response == "":
            try:
                start_time = time.time()
                time_local = time.localtime(start_time)
                l_start = time.strftime("%Y-%m-%d %H:%M:%S", time_local)

                if self.use_wandb:
                    if (start_time - vars.global_start_time) // vars.global_interval > vars.global_steps:
                        if vars.lock.acquire(False):
                            vars.global_steps += 1
                            wandb.log(
                                data={
                                    "Real-time Traffic": vars.global_k_tokens - vars.global_last_tokens_record,
                                    "Total Traffic": vars.global_k_tokens,
                                    "Finished Users": vars.global_finished_users,
                                    "Finished Pages": vars.global_finished_pages,
                                    "Error Cast": vars.global_error_cast / 1000,
                                },
                                step=vars.global_steps,
                            )
                            vars.global_last_tokens_record = vars.global_k_tokens
                            vars.lock.release()

                completion = openai.ChatCompletion.create(
                    model=_resolve_chat_model(self.args),
                    messages=messages,
                    temperature=0.2,
                    request_timeout=timeout,
                    max_tokens=1000,
                )

                l_end = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                k_tokens = completion["usage"]["total_tokens"] / 1000
                print(f"User {self.avatar_id} used {k_tokens} tokens from {l_start} to {l_end}")
                self.memory.user_k_tokens += k_tokens
                vars.global_k_tokens += k_tokens
                response = completion["choices"][0]["message"]["content"]

            except Exception as e:
                vars.global_error_cast += 1
                time.sleep(current_sleep_time)
                if except_waiting_time < max_waiting_time:
                    except_waiting_time *= 2
                current_sleep_time = np.random.randint(0, except_waiting_time - 1)

        return response

    def make_next_decision(self, remember=False, current_page=None):
        observation = "Do you satisfy with current recommendation system and what's your interaction history?"
        relevant_memories = self.memory.fetch_memories(observation)
        formated_relevant_memories = self.memory.format_memories_detail(relevant_memories)

        # Build system prompt with (possibly ablated) activity trait only
        sys_prompt = (
            "You excel at role-playing. Picture yourself as a user exploring a movie recommendation system. "
            "You have the following characteristics:"
            + self._build_profile_text(
                include_activity=True,
                include_conformity=False,
                include_diversity=False,
                include_taste=False,
                include_rating_tendency=False,
            )
            + f"\nNow you are in Page {current_page}. You may get tired with the increase of the pages you have browsed. (above 2 pages is a little bit tired, above 4 pages is very tired)"
            + "\nRelevant context from your memory:"
            + f"\n{formated_relevant_memories}"
        )

        # Make instruction adaptive: don't mention activity if it's ablated
        if self._include_activity():
            feeling_line = (
                "Firstly, generate an overall feeling based on your memory, in accordance with your activity trait and your satisfaction on recommender system."
            )
            fatigue_line = "Next, assess your level of fatigue. You may become tired more easily if you have an inactive activity trait."
            decision_line = "Now, decide whether to continue browsing or exit the recommendation system based on your overall feeling, activity trait, and tiredness."
            exit_line = "You will exit the recommender system either you have negative feelings or you are tired, especially if you have a low activity trait."
        else:
            feeling_line = "Firstly, generate an overall feeling based on your memory and your satisfaction on recommender system."
            fatigue_line = "Next, assess your level of fatigue based on how many pages you have browsed."
            decision_line = "Now, decide whether to continue browsing or exit the recommendation system based on your overall feeling and tiredness."
            exit_line = "You will exit the recommender system either you have negative feelings or you are tired."

        prompt = (
            feeling_line
            + "\nIf your overall feeling is positive, write: POSITIVE: [reason]"
            + "\nIf it's negative, write: NEGATIVE: [reason]"
            + "\n"
            + fatigue_line
            + "\n"
            + decision_line
            + "\n"
            + exit_line
            + "\nTo leave, write: [EXIT]; Reason: [brief reason]"
            + "\nTo continue browsing, write: [NEXT]; Reason: [brief reason]"
        )
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]

        self.write_log("\n" + sys_prompt, color="blue")
        self.write_log("\n" + prompt, color="blue")
        response = self._reaction(messages)
        self.write_log("\n" + response, color="white")

        return response

    def response_to_question(self, question, remember=False):
        relevant_memories = self.memory.memory_retriever.memory_stream
        formated_relevant_memories = self.memory.format_memories_detail(relevant_memories)

        sys_prompt = (
            f"You excel at role-playing. Picture yourself as user {self.avatar_id} who has just finished exploring a movie recommendation system. "
            "You have the following characteristics:"
            + self._build_profile_text(
                include_activity=True,
                include_conformity=True,
                include_diversity=True,
                include_taste=True,
                include_rating_tendency=False,  # keep consistent with original interview prompt
            )
        )

        prompt = f"""
        Relevant context from user {self.avatar_id}'s memory:
        {formated_relevant_memories}
        Act as user {self.avatar_id}, assume you are having a interview, reponse the following question:
        {question}
        """

        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]

        self.write_log("\n" + sys_prompt, color="blue")
        self.write_log("\n" + prompt, color="blue")
        response = self._reaction(messages)
        self.write_log("\n" + response, color="blue")
        if remember:
            self.memory.add_memory(f"I was asked '{question}', and I responsed: '{response}'", now=datetime.datetime.now())
        return response

    def reaction_to_forced_items(self, recommended_items_str):
        # Keep forced_items prompt minimal; it already only uses tastes.
        # Apply ablation: if tastes are removed, don't mention them.
        sys_prompt = "Assume you are a user browsing movie recommendation system."
        if self._include_taste():
            sys_prompt += f"\nYour movie tastes are: {'; '.join(self.taste).replace('I ','')}. "

        prompt = (
            "##recommended list## \n"
            + recommended_items_str
            + "\nPlease choose movies in the ##recommended list## that you want to watch and explain why. After watching the movie, evaluate each movie based on your characteristics and historical ratings to give a rating from 1 to 5."
            + "\nYou only watch movies which align with your preferences."
            + "\nUse this format: MOVIE: [movie name]; WATCH: [yes or no]; REASON: [brief reason]"
            "\nYou must judge all the movies. If you don't want to watch a movie, use WATCH: no; REASON: [brief reason]"
            + "\nEach response should be on one line. Do not include any additional information or explanations and stay grounded in reality."
        )
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
        reaction = self._reaction(messages, timeout=60)
        return reaction

    def reaction_to_recommended_items(self, recommended_items_str, current_page):
        # high_rating text
        try:
            high_rating = self.high_rating.replace("You are", "")
        except Exception:
            high_rating = ""

        # System prompt: include full profile parts (subject to ablation)
        sys_prompt = (
            "You excel at role-playing. Picture yourself as a user exploring a movie recommendation system. "
            "You have the following characteristics:"
            + self._build_profile_text(
                include_activity=True,
                include_conformity=True,
                include_diversity=True,
                include_taste=True,
                include_rating_tendency=True,
            )
        )

        if self.memory.memory_retriever.memory_stream:
            observation = "What movies have you watched on the previous pages of the current recommender system?"
            relevant_memories = self.memory.fetch_memories(observation)
            formated_relevant_memories = self.memory.format_memories_detail(relevant_memories)
            sys_prompt = sys_prompt + f"\nRelevant context from your memory:{formated_relevant_memories}"

        # Adaptive instruction text (do not mention ablated traits)
        # ALIGN basis
        align_basis = "your taste" if self._include_taste() else "your preferences"

        # WATCH basis
        watch_basis = []
        if self._include_activity():
            watch_basis.append("activity trait")
        if self._include_diversity():
            watch_basis.append("diversity trait")
        if watch_basis:
            watch_basis_text = " and ".join(watch_basis)
        else:
            watch_basis_text = "your willingness and selectiveness"

        # RATING basis
        if self._include_conformity():
            rating_basis_text = "your feeling and conformity trait"
        else:
            rating_basis_text = "your feeling"

        prompt = (
            "#### Recommended List #### \n"
            + f"PAGE {current_page}\n"
            + recommended_items_str
            + "\nPlease respond to all the movies in the ## Recommended List ## and provide explanations."
            + f"\nFirstly, determine which movies align with {align_basis} and which do not, and provide reasons. You must respond to all the recommended movies using this format:"
            + "\nMOVIE: [movie name]; ALIGN: [yes or no]; REASON: [brief reason]"
            + f"\nSecondly, among the movies that align with {align_basis}, decide the number of movies you want to watch based on your {watch_basis_text}. Use this format:"
            + "\nNUM: [number of movie you choose to watch]; WATCH: [all movie name you choose to watch]; REASON: [brief reason];"
            + f"\nThirdly, assume it's your first time watching the movies you've chosen, and rate them on a scale of 1-5 to reflect different degrees of liking, considering {rating_basis_text}. Use this format:"
            + "\n MOVIE:[movie you choose to watch]; RATING: [integer between 1-5]; FEELING: [aftermath sentence]; "
            + "\n Do not include any additional information or explanations and stay grounded."
        )

        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]

        self.write_log("\n" + sys_prompt, color="blue")
        self.write_log("\n" + prompt, color="blue")
        reaction = self._reaction(messages, timeout=60)
        self.write_log("\n" + reaction, color="yellow")

        pattern1 = re.compile(r"MOVIE: (.+?); RATING: (\d+); FEELING: (.*)")
        match1 = pattern1.findall(reaction)
        pattern2 = re.compile(r"MOVIE: (.+?); ALIGN: (.+?); REASON: (.*)")
        match2 = pattern2.findall(reaction)
        all_movies = ", ".join([movie_title.strip(";") for movie_title, align, reason in match2])
        watched_movies = [movie_title.strip(";") for movie_title, rating, feeling in match1]
        watched_movies_ratings = [rating.strip(";") for movie_title, rating, feeling in match1]
        dislike_movies = [movie_title.strip(";") for movie_title, rating, feeling in match1 if (int(rating.strip(";")) < 4)]
        dislike_movies.extend([movie_title.strip(";") for movie_title, align, reason in match2 if align.strip(";").lower() == "no"])

        self.memory.add_memory(
            f"The recommender recommended the following movies to me on page {current_page}: {all_movies}, among them, I watched {watched_movies} and rate them {watched_movies_ratings} respectively. I dislike the rest movies: {dislike_movies}.",
            now=datetime.datetime.now(),
        )

        next_decision = self.make_next_decision(current_page=current_page)
        if "[EXIT]" in next_decision or "[exit]" in next_decision:
            self.exit_flag = True
            self.memory.add_memory(
                f"After browsing {current_page} pages, I decided to leave the recommendation system.",
                now=datetime.datetime.now(),
            )
        else:
            self.memory.add_memory(
                f"Turn to page {current_page+1} of the recommendation.",
                now=datetime.datetime.now(),
            )

        return reaction

    def write_log(self, log, color=None, attrs=None, print=False):
        with open(self.log_file, "a") as f:
            f.write(log + "\n")
            f.flush()
        if print:
            cprint(log, color=color, attrs=attrs)