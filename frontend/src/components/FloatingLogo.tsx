import React from "react";

export interface FloatingLogoProps {
  src?: string;
  alt?: string;
  href?: string;
  onClick?: () => void;
  size?: number;
  offset?: { x?: number; y?: number };
  className?: string;
  tooltip?: string;
  hideOn?: Array<"mobile" | "print">;
}

const FloatingLogo: React.FC<FloatingLogoProps> = ({
  src = "/logo.png",
  alt = "FinBuddy",
  href,
  onClick,
  size = 56,
  offset = { x: 16, y: 16 },
  className = "",
  tooltip,
  hideOn = [],
}) => {
  const style: React.CSSProperties = {
    width: size,
    height: size,
    right: `calc(${offset.x ?? 16}px + env(safe-area-inset-right, 0px))`,
    bottom: `calc(${offset.y ?? 16}px + env(safe-area-inset-bottom, 0px))`,
  };

  const base = [
    "fixed z-40",
    hideOn.includes("print") ? "print:hidden" : "",
    hideOn.includes("mobile") ? "hidden sm:flex" : "flex",
    "items-center justify-center",
    "rounded-full bg-white/95 dark:bg-zinc-900/90",
    "shadow-lg ring-1 ring-black/10 dark:ring-white/10",
    "backdrop-blur supports-[backdrop-filter]:bg-white/70",
    "transition active:scale-95",
    className,
  ].filter(Boolean).join(" ");

  const img = (
    <img
      src={src}
      alt={alt}
      draggable={false}
      className="h-3/5 w-3/5 object-contain select-none"
      aria-hidden={!!(href || onClick)}
    />
  );

  return href ? (
    <a href={href} aria-label={alt} title={tooltip || alt} className={base} style={style}>
      {img}
    </a>
  ) : (
    <button type="button" aria-label={alt} title={tooltip || alt} onClick={onClick} className={base} style={style}>
      {img}
    </button>
  );
};

export default FloatingLogo;
