import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

class AlternativeDataFetcher:
    def __init__(self):
        self.oecd_base_url = "https://stats.oecd.org/restsdmx/sdmx.ashx/GetData"
        self.imf_base_url = "http://dataservices.imf.org/REST/SDMX_JSON.svc"
        self.eurostat_base_url = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1"
        
    def fetch_oecd_data(self, dataset: str, countries: List[str], start_year: int, end_year: int) -> pd.DataFrame:
        """
        Fetch data from OECD API
        
        Parameters:
        -----------
        dataset : str
            OECD dataset identifier
        countries : List[str]
            List of country codes
        start_year : int
            Start year for data collection
        end_year : int
            End year for data collection
        """
        data_frames = []
        for country in countries:
            url = f"{self.oecd_base_url}/{dataset}/{country}/all"
            params = {
                'startTime': start_year,
                'endTime': end_year,
                'format': 'json'
            }
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                # Process OECD JSON response
                df = pd.DataFrame(data['dataSets'][0]['observations']).T
                df.columns = [col['name'] for col in data['structure']['dimensions']['observation']]
                data_frames.append(df)
                
            except Exception as e:
                logger.error(f"Error fetching OECD data for {country}: {str(e)}")
                continue
                
        return pd.concat(data_frames) if data_frames else pd.DataFrame()
    
    def fetch_imf_data(self, dataset: str, countries: List[str], indicators: List[str]) -> pd.DataFrame:
        """
        Fetch data from IMF API
        
        Parameters:
        -----------
        dataset : str
            IMF dataset identifier
        countries : List[str]
            List of country codes
        indicators : List[str]
            List of indicator codes
        """
        data_frames = []
        for country in countries:
            for indicator in indicators:
                url = f"{self.imf_base_url}/CompactData/{dataset}"
                params = {
                    'countries': country,
                    'indicators': indicator
                }
                try:
                    response = requests.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    
                    # Process IMF JSON response
                    series = data['CompactData']['DataSet']['Series']
                    df = pd.DataFrame(series['Obs'])
                    df['Country'] = country
                    df['Indicator'] = indicator
                    data_frames.append(df)
                    
                except Exception as e:
                    logger.error(f"Error fetching IMF data for {country}, {indicator}: {str(e)}")
                    continue
                    
        return pd.concat(data_frames) if data_frames else pd.DataFrame()
    
    def fetch_eurostat_data(self, dataset: str, countries: List[str], start_period: str, end_period: str) -> pd.DataFrame:
        """
        Fetch data from Eurostat API
        
        Parameters:
        -----------
        dataset : str
            Eurostat dataset identifier
        countries : List[str]
            List of country codes
        start_period : str
            Start period in format 'YYYY-MM'
        end_period : str
            End period in format 'YYYY-MM'
        """
        url = f"{self.eurostat_base_url}/data/{dataset}"
        params = {
            'format': 'JSON',
            'geo': ','.join(countries),
            'startPeriod': start_period,
            'endPeriod': end_period
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Process Eurostat JSON response
            df = pd.DataFrame(data['value'])
            df['time'] = pd.to_datetime(df['time'])
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Eurostat data: {str(e)}")
            return pd.DataFrame()

class RealTimeDataFetcher:
    def __init__(self):
        self.exchange_rate_url = "https://api.exchangerate-api.com/v4/latest"
        self.commodity_price_url = "https://api.commodities-api.com/api/latest"
        
    def fetch_exchange_rates(self, base_currency: str, target_currencies: List[str]) -> Dict[str, float]:
        """
        Fetch real-time exchange rates
        
        Parameters:
        -----------
        base_currency : str
            Base currency code (e.g., 'USD')
        target_currencies : List[str]
            List of target currency codes
        """
        try:
            response = requests.get(f"{self.exchange_rate_url}/{base_currency}")
            response.raise_for_status()
            data = response.json()
            
            rates = {currency: data['rates'][currency] 
                    for currency in target_currencies 
                    if currency in data['rates']}
            
            return {
                'timestamp': data['time_last_updated'],
                'base_currency': base_currency,
                'rates': rates
            }
            
        except Exception as e:
            logger.error(f"Error fetching exchange rates: {str(e)}")
            return {}
    
    def fetch_commodity_prices(self, commodities: List[str]) -> Dict[str, float]:
        """
        Fetch real-time commodity prices
        
        Parameters:
        -----------
        commodities : List[str]
            List of commodity codes
        """
        try:
            response = requests.get(self.commodity_price_url)
            response.raise_for_status()
            data = response.json()
            
            prices = {commodity: data['data']['rates'][commodity] 
                     for commodity in commodities 
                     if commodity in data['data']['rates']}
            
            return {
                'timestamp': data['data']['timestamp'],
                'base': data['data']['base'],
                'prices': prices
            }
            
        except Exception as e:
            logger.error(f"Error fetching commodity prices: {str(e)}")
            return {} 