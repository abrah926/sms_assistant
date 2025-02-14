#!/usr/bin/env python3
import os
import subprocess
import sys
from datetime import datetime
import shutil

def backup_env_file():
    """Create a backup of the .env file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = f'.env.backup_{timestamp}'
    try:
        shutil.copy2('.env', backup_file)
        print(f"Created backup: {backup_file}")
        return True
    except Exception as e:
        print(f"Warning: Could not create backup: {e}", file=sys.stderr)
        return False

def get_azure_token():
    """Get Azure access token"""
    try:
        result = subprocess.run(
            ['az', 'account', 'get-access-token', 
             '--resource', 'https://ossrdbms-aad.database.windows.net',
             '--query', 'accessToken',
             '--output', 'tsv'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error getting Azure token: {e}", file=sys.stderr)
        print("Make sure you're logged in with 'az login'", file=sys.stderr)
        sys.exit(1)

def update_env_file(token):
    """Update the PGPASSWORD in .env file"""
    if not os.path.exists('.env'):
        print("Error: .env file not found", file=sys.stderr)
        sys.exit(1)

    backup_env_file()

    try:
        with open('.env', 'r') as f:
            lines = f.readlines()
        
        # Update or add PostgreSQL configuration
        pg_config = {
            'PGHOST': 'sms1758.postgres.database.azure.com',
            'PGUSER': 'avidal491@live.edpuniversity.edu',
            'PGDATABASE': 'postgres',
            'PGPASSWORD': token
        }
        
        # Remove existing PG* lines
        lines = [line for line in lines if not any(line.startswith(key) for key in pg_config.keys())]
        
        # Add new PG* configuration at the top
        pg_lines = [f'{key}={value}\n' for key, value in pg_config.items()]
        lines = pg_lines + lines
        
        with open('.env', 'w') as f:
            f.writelines(lines)
            
    except Exception as e:
        print(f"Error updating .env file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    print("Getting Azure access token...")
    token = get_azure_token()
    print("Updating .env file with new token...")
    update_env_file(token)
    print("Done! You can now run your database commands.")