#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <limits>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    void check_status(const arrow::Status & status)
    {
        if (!status.ok())
        {
            throw std::runtime_error(status.ToString());
        }
    }
    std::int32_t date_days(const std::string & value)
    {
        if (value.size() != 10 || value[4] != '-' || value[7] != '-')
            throw std::runtime_error("Expected date YYYY-MM-DD: " + value);

        for (std::size_t i = 0; i < value.size(); ++i)
        {
            if (i != 4 && i != 7 && (value[i] < '0' || value[i] > '9'))
                throw std::runtime_error("Invalid date: " + value);
        }

        const int year = std::stoi(value.substr(0, 4));
        const unsigned month = std::stoul(value.substr(5, 2));
        const unsigned day = std::stoul(value.substr(8, 2));
        const std::chrono::year_month_day date{
            std::chrono::year{year}, std::chrono::month{month}, std::chrono::day{day}};

        if (year < 1 || !date.ok())
            throw std::runtime_error("Invalid calendar date: " + value);

        return static_cast<std::int32_t>(
            std::chrono::sys_days{date}.time_since_epoch().count());
    }

    void save(const std::filesystem::path & path, const std::vector<std::string> & names, const std::vector<nlohmann::json> & rows)
    {
        std::vector<std::shared_ptr<arrow::Field>> fields;
        std::vector<std::shared_ptr<arrow::Array>> columns;

        for (std::size_t i = 0; i < names.size(); ++i)
        {
            std::shared_ptr<arrow::DataType> type = arrow::utf8();

            if (names[i] == "profile_index" || names[i] == "flight_index" || names[i] == "card_index")
                type = arrow::int64();
            else if (names[i] == "codeshare")
                type = arrow::boolean();
            else if (names[i] == "date")
                type = arrow::date32();

            std::unique_ptr<arrow::ArrayBuilder> builder;
            check_status(arrow::MakeBuilder(arrow::default_memory_pool(), type, &builder));

            for (const auto & row : rows)
            {
                const auto & value = row.at(i);

                if (value.is_null())
                {
                    check_status(builder->AppendNull());
                }
                else if (type->id() == arrow::Type::INT64)
                {
                    if (!value.is_number_integer() ||
                        (value.is_number_unsigned() && value.get<std::uint64_t>() >
                         static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())))
                        throw std::runtime_error("Expected integer index: " + names[i]);

                    const auto index = value.get<std::int64_t>();
                    if (index < 0)
                        throw std::runtime_error("Negative index: " + names[i]);

                    check_status(static_cast<arrow::Int64Builder &>(*builder).Append(index));
                }
                else if (type->id() == arrow::Type::BOOL)
                {
                    if (!value.is_boolean())
                        throw std::runtime_error("Expected JSON boolean: " + names[i]);

                    check_status(static_cast<arrow::BooleanBuilder &>(*builder).Append(value.get<bool>()));
                }
                else if (type->id() == arrow::Type::DATE32)
                {
                    if (!value.is_string())
                        throw std::runtime_error("Expected ISO date string: " + names[i]);

                    check_status(static_cast<arrow::Date32Builder &>(*builder)
                        .Append(date_days(value.get<std::string>())));
                }
                else
                {
                    if (!value.is_string())
                        throw std::runtime_error("Expected text, refusing numeric conversion: " + names[i]);

                    check_status(static_cast<arrow::StringBuilder &>(*builder).Append(value.get<std::string>()));
                }
            }

            std::shared_ptr<arrow::Array> column;
            check_status(builder->Finish(&column));

            fields.push_back(arrow::field(names[i], type));
            columns.push_back(column);
        }

        auto table = arrow::Table::Make(arrow::schema(fields), columns);
        check_status(table->ValidateFull());

        auto result = arrow::io::FileOutputStream::Open(path.string());

        if (!result.ok())
        {
            throw std::runtime_error(result.status().ToString());
        }

        const auto & output = result.ValueOrDie();

        check_status(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), output, 50'000));

        check_status(output->Close());
    }
}

int main(int argc, char * argv[])
{
    try
    {
        if (argc != 3)
        {
            throw std::runtime_error("Usage: nlohmann::json_learning input.nlohmann::json output_directory");
        }

        std::ifstream input(argv[1]);
        if (!input)
        {
            throw std::runtime_error("Cannot open input nlohmann::json");
        }

        const auto root = nlohmann::json::parse(input);
        const auto& profiles = root.at("Forum Profiles");

        if (!profiles.is_array())
            throw std::runtime_error("Forum Profiles must be an array");

        std::vector<nlohmann::json> people;
        std::vector<nlohmann::json> flights;
        std::vector<nlohmann::json> cards;

        for (std::size_t p = 0; p < profiles.size(); ++p)
        {
            const auto & person = profiles.at(p);
            const auto & name = person.at("Real Name");

            people.push_back(nlohmann::json::array({
                p, person.at("NickName"), person.at("Sex"),
                name.at("First Name"), name.at("Last Name"),
                person.at("Travel Documents").dump()
            }));

            const auto & trips = person.at("Registered Flights");

            if (!trips.is_array())
                throw std::runtime_error("Flights must be an array");

            for (std::size_t f = 0; f < trips.size(); ++f)
            {
                const auto & trip = trips.at(f);
                const auto & from = trip.at("Departure");
                const auto & to = trip.at("Arrival");

                flights.push_back(nlohmann::json::array({
                    p, f, trip.at("Date"), trip.at("Flight"),
                    trip.at("Codeshare"),
                    from.at("City"), from.at("Airport"), from.at("Country"),
                    to.at("City"), to.at("Airport"), to.at("Country")
                }));
            }

            const auto & loyalty = person.at("Loyality Programm");

            if (!loyalty.is_array())
            {
                throw std::runtime_error("Cards must be an array");
            }

            for (std::size_t c = 0; c < loyalty.size(); ++c)
            {
                const auto& card = loyalty.at(c);

                cards.push_back(nlohmann::json::array({
                    p, c, card.at("programm"),
                    card.at("Number"), card.at("Status")
                }));
            }
        }

        const std::filesystem::path destination = argv[2];
        const std::filesystem::path temporary = destination.string() + ".partial";

        if (std::filesystem::exists(destination) || std::filesystem::exists(temporary))
        {
            throw std::runtime_error("Output directory already exists");
        }

        std::filesystem::create_directories(temporary);

        save(temporary / "profiles.parquet",
{
            "profile_index", "nickname", "sex",
            "first_name", "last_name", "travel_documents_nlohmann::json"
        }, people);

        save(temporary / "flights.parquet",
{
            "profile_index", "flight_index", "date", "flight", "codeshare",
            "departure_city", "departure_airport", "departure_country",
            "arrival_city", "arrival_airport", "arrival_country"
        }, flights);

        save(temporary / "cards.parquet",
{
            "profile_index", "card_index", "program", "number", "status"
        }, cards);

        std::filesystem::rename(temporary, destination);

        std::cout << "Profiles: " << people.size() << '\n';
        std::cout << "Flights: " << flights.size() << '\n';
        std::cout << "Cards: " << cards.size() << '\n';
    }
    catch (const std::exception & e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}