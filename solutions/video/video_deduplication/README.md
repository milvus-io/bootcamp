# Video Deduplication

Video Deduplication, also known as Video Copy Detection or Video Identification by Fingerprinting, means that given a query video, you need to find or retrieval the videos with the same content with query video.

Due to the popularity of Internet-based video sharing services, the volume of video content on the Web has reached unprecedented scales. Besides copyright protection, a video copy detection system is important in applications like video classification, tracking, filtering and recommendation.  

Generally speaking, video deduplication tasks can be divided into two categories according to the retrieval level: one is video-level deduplication, and the other is segment-level deduplication . 

- Video-level deduplication is a method for situations with high repetition. It finds duplicate videos by comparing the similarity between the embeddings of the whole video. Since only one embedding is extracted from a video, this method works faster. But the limitation of this method is also obvious: it is not good for detecting similar videos of different lengths. For example, the first quarter of video A and video B are exactly the same, but their embeddings may not be similar. In this case, it is obviously impossible to detect infringing content.

- Segment-level deduplication detects the specific start and end times of repeated segments, which can handle complex clipping and insertion of video segments as well as situations where the video lengths are not equal. It does so by comparing the similarity between video frames. Obviously, we need to use this method in the actual task of mass video duplication checking. Of course, the speed of this method will be slower than the one of video level.

## Learn from Notebook

- [Getting started with a video-level example](https://github.com/towhee-io/examples/tree/main/video/video_deduplicationvideo_level/video_deduplication_at_video_level.ipynb)

In this notebook you will get prerequisites, build and use a basic Video Deduplication system based on video level, visualize sample results, and measure the system with performance metrics.

- [Build a practical segment-level example](https://github.com/towhee-io/examples/tree/main/video/video_deduplication/segment_level/video_deduplication_at_segment_level.ipynb)

In this notebook you will get prerequisites, build a more practical Video Deduplication system with greater robustness, more engineered solution, and finer-grained results.
