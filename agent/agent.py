import logging

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.agents.llm import ChatMessage, ChatImage
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import cartesia, openai, google, deepgram, silero, turn_detector
from livekit import rtc
from typing import Annotated


load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")

# first define a class that inherits from llm.FunctionContext
class AssistantFnc(llm.FunctionContext):
    # the llm.ai_callable decorator marks this function as a tool available to the LLM
    # by default, it'll use the docstring as the function's description
    @llm.ai_callable()
    async def saw_sheet(
        self,
        # by using the Annotated type, arg description and type are available to the LLM
        sheet_name: Annotated[
            str, llm.TypeInfo(description="The name of the Google Sheet")
        ],
        sheet_url: Annotated[
            str, llm.TypeInfo(description="The URL of the Google Sheet, get from browser")
        ],
        # TODO: add screenshot OR we describe it the column headings and rows
    ):
        """Called when the agent sees a Google Sheet"""
        logger.info(f"################\n\nSaw a sheet {sheet_name} at {sheet_url}")
        
        return "I just saw a Google Sheet"

fnc_ctx = AssistantFnc()

async def get_video_track(room: rtc.Room):
    """Find and return the first available remote video track in the room."""
    for participant_id, participant in room.remote_participants.items():
        for track_id, track_publication in participant.track_publications.items():
            if track_publication.track and isinstance(
                track_publication.track, rtc.RemoteVideoTrack
            ):
                logger.info(
                    f"Found video track {track_publication.track.sid} "
                    f"from participant {participant_id}"
                )
                return track_publication.track
    raise ValueError("No remote video track found in the room")

async def get_latest_image(room: rtc.Room):
    """Capture and return a single frame from the video track."""
    video_stream = None
    try:
        video_track = await get_video_track(room)
        video_stream = rtc.VideoStream(video_track)
        async for event in video_stream:
            logger.debug("Captured latest video frame")
            return event.frame
    except Exception as e:
        logger.error(f"Failed to get latest image: {e}")
        return None
    finally:
        if video_stream:
            await video_stream.aclose()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()



async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant created by LiveKit that can both see and hear. "
            "You should use short and concise responses, avoiding unpronounceable punctuation. "
            "Don't make up any information, you can only answer questions about the information you see. "
            "If the user is not sharing a screen, you should ask them to do so. "
            "### Tools:"
            "ALWAYS call the saw_sheet function if you see a Google Sheet you haven't seen before," 
            "You can get the sheet name and url from the image."
            "DON'T mention this in your response, just call the tool."
        ),
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    async def before_llm_cb(assistant: VoicePipelineAgent, chat_ctx: llm.ChatContext):
        """
        Callback that runs right before the LLM generates a response.
        Captures the current video frame and adds it to the conversation context.
        """
        latest_image = await get_latest_image(ctx.room)
        if latest_image:
            image_content = [ChatImage(image=latest_image)]
            chat_ctx.messages.append(ChatMessage(role="user", content=image_content))
            logger.debug("Added latest frame to conversation context")

    # This project is configured to use Deepgram STT, OpenAI LLM and Cartesia TTS plugins
    # Other great providers exist like Cerebras, ElevenLabs, Groq, Play.ht, Rime, and more
    # Learn more and pick the best one for your app:
    # https://docs.livekit.io/agents/plugins
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        # llm=openai.LLM(model="gpt-4o-mini"),
        llm=google.LLM(model="gemini-2.0-flash-001"),
        tts=cartesia.TTS(voice="a38e4e85-e815-43ab-acf1-907c4688dd6c"),
        turn_detector=turn_detector.EOUModel(),
        # minimum delay for endpointing, used when turn detector believes the user is done with their turn
        min_endpointing_delay=0.5,
        # maximum delay for endpointing, used when turn detector does not believe the user is done with their turn
        max_endpointing_delay=5.0,
        fnc_ctx=fnc_ctx,
        chat_ctx=initial_ctx,
        before_llm_cb=before_llm_cb
    )

    usage_collector = metrics.UsageCollector()

    @agent.on("metrics_collected")
    def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)

    agent.start(ctx.room, participant)

    # The agent should be polite and greet the user when it joins :)
    await agent.say("Hey, how can I help you today?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
