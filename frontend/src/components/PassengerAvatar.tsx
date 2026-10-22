const PLACEHOLDER_PATH = '/passenger-placeholder.png';

export default function PassengerAvatar({ className = '' }: { className?: string }) {
  const classes = ['passenger-avatar', className].filter(Boolean).join(' ');

  return (
    <span className={classes} aria-hidden="true">
      <img src={PLACEHOLDER_PATH} alt="" />
    </span>
  );
}
