#include <xlnt/xlnt.hpp>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    void check_status(const arrow::Status& status)
    {
        if (!status.ok())
            throw std::runtime_error(status.ToString());
    }

    std::string read_value(const xlnt::cell & cell, const std::string & name)
    {
        const auto address = cell.reference().to_string();

        if (cell.has_formula() || cell.data_type() == xlnt::cell_type::error)
            throw std::runtime_error("Formula or Excel error at " + address);

        if (name == "PassengerBirthDate" || name == "FlightDate")
        {
            if (cell.data_type() != xlnt::cell_type::number || !cell.is_date())
                throw std::runtime_error("Expected Excel date at " + address);

            const double value = cell.value<double>();

            if (!std::isfinite(value) || value < 1 ||
                value > 2'958'465 || std::floor(value) != value)
                throw std::runtime_error("Invalid Excel date at " + address);

            const auto date = xlnt::date::from_number(
                static_cast<int>(value), cell.base_date());

            return std::format("{:04}-{:02}-{:02}", date.year, date.month, date.day);
        }

        if (name == "FlightTime")
        {
            if (cell.data_type() != xlnt::cell_type::number)
            {
                throw std::runtime_error("Expected Excel time at " + address);
            }

            const double value = cell.value<double>();

            if (!std::isfinite(value) || value < 0 || value >= 1)
            {
                throw std::runtime_error("Invalid Excel time at " + address);
            }

            const auto minutes = std::llround(value * 1440);

            if (minutes >= 1440 || std::abs(value * 86400 - static_cast<double>(minutes) * 60) > 0.001)
            {
                throw std::runtime_error("Expected minute precision at " + address);
            }

            return std::format("{:02}:{:02}", minutes / 60, minutes % 60);
        }

        if (cell.data_type() != xlnt::cell_type::shared_string &&
        cell.data_type() != xlnt::cell_type::inline_string)
        {
            throw std::runtime_error("Expected text at " + address);
        }

        return cell.value<std::string>();
    }
}


int main(int argc, char* argv[])
{
    try
    {
        if (argc != 3)
        {
            throw std::runtime_error("Usage: csv_in_excel_to_parquet input.xlsx output.parquet");
        }


        const std::filesystem::path destination = argv[2];
        const std::filesystem::path temporary = destination.string() + ".partial";

        if (std::filesystem::exists(destination) || std::filesystem::exists(temporary))
        {
            throw std::runtime_error("Output already exists");
        }

        xlnt::workbook workbook;
        workbook.load(argv[1]);

        auto sheet = workbook.sheet_by_title("BoardingData");

        const std::vector<std::string> names = {
            "PassengerFirstName",
            "PassengerSecondName",
            "PassengerLastName",
            "PassengerSex",
            "PassengerBirthDate",
            "PassengerDocument",
            "BookingCode",
            "TicketNumber",
            "Baggage",
            "FlightDate",
            "FlightTime",
            "FlightNumber",
            "CodeShare",
            "Destination"
        };

        for (std::size_t c = 0; c < names.size(); ++c)
        {
            const auto address = std::string(1, static_cast<char>('A' + c)) + "1";

            const auto header = sheet.cell(address);

            if (!header.has_value() || header.has_formula() || header.to_string() != names[c])
            {
                throw std::runtime_error("Unexpected header at " + address);
            }
        }

        std::vector<std::shared_ptr<arrow::Field>> fields;
        std::vector<std::shared_ptr<arrow::Array>> columns;

        const auto last_row = sheet.highest_row();

        for (std::size_t c = 0; c < names.size(); ++c)
        {
            arrow::StringBuilder builder;

            for (xlnt::row_t r = 2; r <= last_row; ++r)
            {
                const auto address = std::string(1, static_cast<char>('A' + c)) + std::to_string(r);

                const auto cell = sheet.cell(address);

                if (cell.has_formula() || cell.data_type() == xlnt::cell_type::error)
                {
                    throw std::runtime_error("Formula or Excel error at " + address);
                }

                if (!cell.has_value())
                {
                    check_status(builder.AppendNull());
                }
                else
                {
                    check_status(builder.Append(read_value(cell, names[c])));
                }
            }

            std::shared_ptr<arrow::Array> column;
            check_status(builder.Finish(&column));

            fields.push_back(arrow::field(names[c], arrow::utf8()));
            columns.push_back(column);
        }

        auto table = arrow::Table::Make(arrow::schema(fields), columns);
        check_status(table->ValidateFull());

        auto result = arrow::io::FileOutputStream::Open(temporary.string());

        if (!result.ok())
        {
            throw std::runtime_error(result.status().ToString());
        }

        const auto & output = result.ValueOrDie();

        auto properties = parquet::WriterProperties::Builder()
            .compression(arrow::Compression::ZSTD)
            ->build();

        check_status(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), output, 50'000, properties));
        check_status(output->Close());

        std::filesystem::rename(temporary, destination);

        std::cout << "Rows: " << table->num_rows() << '\n';
        std::cout << "Columns: " << table->num_columns() << '\n';
        std::cout << "Saved: " << destination.string() << '\n';
    }
    catch (const std::exception & e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }

    return 0;
}