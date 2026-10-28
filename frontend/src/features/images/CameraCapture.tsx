import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "../../components/Button";

type Phase = "starting" | "live" | "captured" | "error";

function describeCameraError(error: unknown): string {
  const name = error instanceof DOMException ? error.name : "";
  switch (name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Camera permission was denied. Allow camera access in your browser settings, or choose photos from your device instead.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "No camera was found on this device. You can choose photos from your device instead.";
    case "NotReadableError":
      return "The camera is in use by another application. Close it and try again.";
    default:
      return "The camera could not be started. You can choose photos from your device instead.";
  }
}

interface Props {
  onCapture: (file: File) => void;
  onClose: () => void;
}

/** Mounted only after the user asks for the camera, so permission is never requested earlier. */
export function CameraCapture({ onCapture, onClose }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [phase, setPhase] = useState<Phase>("starting");
  const [error, setError] = useState<string | null>(null);
  const [captured, setCaptured] = useState<{ file: File; url: string } | null>(null);

  const stopStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }, []);

  useEffect(() => {
    let cancelled = false;
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("This browser cannot access the camera (it may require HTTPS). Choose photos from your device instead.");
      setPhase("error");
      return;
    }
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: "environment" }, audio: false })
      .then((stream) => {
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop()); // closed before permission resolved
          return;
        }
        streamRef.current = stream;
        setPhase("live");
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setError(describeCameraError(e));
        setPhase("error");
      });
    return () => {
      cancelled = true;
      stopStream();
    };
  }, [stopStream]);

  // Attach the stream once the <video> element exists.
  useEffect(() => {
    if (phase === "live" && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
    }
  }, [phase]);

  useEffect(() => () => {
    if (captured) URL.revokeObjectURL(captured.url);
  }, [captured]);

  function capture() {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0);
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          setError("Could not capture the photo. Please try again.");
          return;
        }
        const file = new File([blob], `camera-${Date.now()}.jpg`, { type: "image/jpeg" });
        setCaptured({ file, url: URL.createObjectURL(blob) });
        setPhase("captured");
      },
      "image/jpeg",
      0.92,
    );
  }

  function retake() {
    setCaptured(null);
    setPhase("live");
  }

  function close() {
    stopStream();
    onClose();
  }

  return (
    <div className="space-y-3 rounded-lg border border-slate-200 bg-white p-4" aria-label="Camera">
      {phase === "starting" && <p role="status">Waiting for camera permission…</p>}
      {phase === "error" && (
        <p role="alert" className="text-sm text-red-700">
          {error}
        </p>
      )}
      {phase === "live" && (
        <video ref={videoRef} autoPlay playsInline muted className="w-full rounded-md bg-black" />
      )}
      {phase === "captured" && captured && (
        <img src={captured.url} alt="Captured photo preview" className="w-full rounded-md" />
      )}
      {error && phase !== "error" && (
        <p role="alert" className="text-sm text-red-700">
          {error}
        </p>
      )}
      <div className="flex flex-wrap gap-2">
        {phase === "live" && <Button onClick={capture}>Capture photo</Button>}
        {phase === "captured" && captured && (
          <>
            <Button
              onClick={() => {
                onCapture(captured.file);
                close();
              }}
            >
              Use photo
            </Button>
            <Button variant="secondary" onClick={retake}>
              Retake
            </Button>
          </>
        )}
        <Button variant="secondary" onClick={close}>
          Close camera
        </Button>
      </div>
    </div>
  );
}
