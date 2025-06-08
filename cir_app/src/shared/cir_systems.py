import threading

# Global CIR system instances
cir_system_searle = None
cir_system_freedom = None

# Lock for thread safety
lock = threading.Lock()