use log::{LevelFilter, Log, Metadata, Record};
use std::time::SystemTime;

pub struct SimpleLogger;

impl Log for SimpleLogger {
    fn enabled(&self, _metadata: &Metadata) -> bool {
        // Enable logging for all levels
        true
    }

    fn log(&self, record: &Record) {
        // Only log if enabled for this level/target
        if self.enabled(record.metadata()) {
            let timestamp = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);

            let location = match (record.file(), record.line()) {
                (Some(file), Some(line)) => format!("{}:{}", file, line),
                (Some(file), None) => file.to_string(),
                (None, _) => String::from("unknown location"),
            };

            println!(
                "[{timestamp}][{level}][{target}][{location}] {message}",
                timestamp = timestamp,
                level = record.level(),
                target = record.target(),
                location = location,
                message = record.args()
            );
        }
    }

    fn flush(&self) {}
}

// Initialize the logger
pub fn init() -> Result<(), log::SetLoggerError> {
    static LOGGER: SimpleLogger = SimpleLogger;

    log::set_logger(&LOGGER).map(|()| log::set_max_level(LevelFilter::Trace))
}
