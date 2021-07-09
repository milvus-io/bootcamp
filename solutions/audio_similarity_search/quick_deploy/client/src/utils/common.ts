export const formatDuration = (duration: number) => {
  if (duration) {
    const min = Math.floor(duration / 60);
    const sec = duration % 60;
    let minText = min.toString();
    let secText = sec.toString();
    if (min < 10) {
      minText += "0";
    }
    if (sec < 10) {
      secText = "0" + sec.toFixed(2);
    }
    return `${minText}:${secText}`;
  }
  return "--:--";
};
